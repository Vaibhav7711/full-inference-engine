"""Thread-owned bridge from async HTTP handlers to continuous GPU batching.

Failure model:
    - Request-level problems (invalid input, queue full, capacity exceeded) are surfaced
      per request as ``SubmitError`` at submission time or as a terminal request state.
    - Engine-level problems (an exception escaping ``engine.step``) are fatal for the
      worker: every outstanding handle is completed with the error, ``alive`` turns
      False, and the process is expected to be restarted by its supervisor. A CUDA
      context that has raised cannot be trusted to keep serving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import monotonic
from typing import Callable, Sequence
from uuid import uuid4

from engine.runtime import GenerationRequest


class ServerShutdown(RuntimeError):
    """Completion error for handles abandoned by a stopping worker."""


class SubmitError(RuntimeError):
    """A submission rejected before reaching the scheduler, with an HTTP status hint."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class RequestHandle:
    """A caller-owned view of one scheduler request."""

    request: GenerationRequest
    completed: Event = field(default_factory=Event)
    # Set once the scheduler has taken the request (or rejected it, in which case
    # ``completed`` is set too). Lets streaming responses send real status codes.
    accepted: Event = field(default_factory=Event)
    error: BaseException | None = None
    _callbacks: list[Callable[["RequestHandle"], None]] = field(default_factory=list)
    _accept_callbacks: list[Callable[["RequestHandle"], None]] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock)

    def on_complete(self, callback: Callable[["RequestHandle"], None]) -> None:
        """Run ``callback(handle)`` from the completing thread, or now if already done."""
        with self._lock:
            if not self.completed.is_set():
                self._callbacks.append(callback)
                return
        callback(self)

    def on_accept(self, callback: Callable[["RequestHandle"], None]) -> None:
        """Run ``callback(handle)`` once the scheduler has decided on admission."""
        with self._lock:
            if not self.accepted.is_set():
                self._accept_callbacks.append(callback)
                return
        callback(self)

    def _accept(self) -> None:
        with self._lock:
            if self.accepted.is_set():
                return
            self.accepted.set()
            callbacks, self._accept_callbacks = self._accept_callbacks, []
        for callback in callbacks:
            try:
                callback(self)
            except Exception:
                pass

    def _complete(self, error: BaseException | None = None) -> None:
        with self._lock:
            if self.completed.is_set():
                return
            if error is not None and self.error is None:
                self.error = error
            self.completed.set()
            callbacks, self._callbacks = self._callbacks, []
        for callback in callbacks:
            try:
                callback(self)
            except Exception:  # a misbehaving observer must not break the worker
                pass
        self._accept()  # a decided request is no longer pending admission


class ContinuousBatchingService:
    """Own an engine from one worker thread while API handlers remain asynchronous.

    All scheduler and GPU state is touched exclusively by the worker. Callers only add
    immutable token IDs to the bounded inbox and observe the request lifecycle.
    """

    def __init__(self, engine, *, max_pending_submissions: int = 256) -> None:
        if max_pending_submissions <= 0:
            raise ValueError("max_pending_submissions must be positive")
        self.engine = engine
        self._inbox: Queue[RequestHandle] = Queue(maxsize=max_pending_submissions)
        self._cancellations: Queue[tuple[str, str]] = Queue()
        self._active: dict[str, RequestHandle] = {}
        self._lock = Lock()
        self._wake = Event()
        self._stopping = Event()
        self._draining = Event()
        self._thread: Thread | None = None
        self._fatal_error: BaseException | None = None
        self._completed_count = 0
        self._cancelled_count = 0
        self._failed_count = 0

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = Thread(target=self._worker, name="continuous-batching-gpu", daemon=True)
        self._thread.start()

    def drain(self, timeout_s: float = 30.0) -> int:
        """Stop admitting, let in-flight work finish, then cancel whatever remains.

        Returns the number of requests cancelled because the deadline passed.
        """
        self._draining.set()
        self._wake.set()
        deadline = monotonic() + max(0.0, timeout_s)
        while monotonic() < deadline and self._thread is not None and self._thread.is_alive():
            with self._lock:
                outstanding = bool(self._active) or not self._inbox.empty()
            if not outstanding:
                break
            self._wake.set()
            self._pause(0.05)
        with self._lock:
            leftovers = list(self._active.values())
        for handle in leftovers:
            self.cancel(handle, reason="SERVER_SHUTDOWN")
        if leftovers:
            # Give the worker one more pass to apply the cancellations before stopping.
            self._wake.set()
            self._pause(0.2)
        return len(leftovers)

    def stop(self, *, drain_timeout_s: float = 30.0) -> None:
        """Drain, stop the worker, and complete every handle that is still waiting."""
        if self._thread is not None and self._thread.is_alive():
            self.drain(drain_timeout_s)
        self._stopping.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        self._fail_all(ServerShutdown("server shutdown"), mark_fatal=False)

    @staticmethod
    def _pause(seconds: float) -> None:
        Event().wait(seconds)

    # ------------------------------------------------------------------ status
    @property
    def alive(self) -> bool:
        """Liveness: the worker thread exists and has not hit an engine-level error."""
        return (
            self._fatal_error is None
            and self._thread is not None
            and self._thread.is_alive()
        )

    @property
    def ready(self) -> bool:
        """Readiness: alive and accepting new work."""
        return self.alive and not self._draining.is_set() and not self._stopping.is_set()

    @property
    def fatal_error(self) -> BaseException | None:
        return self._fatal_error

    # ------------------------------------------------------------------ submission
    def submit(self, prompt_token_ids: Sequence[int], max_new_tokens: int) -> RequestHandle:
        if self._fatal_error is not None:
            raise SubmitError("engine worker is unavailable", status_code=503)
        if self._thread is not None and not self._thread.is_alive():
            raise SubmitError("engine worker has stopped", status_code=503)
        if self._draining.is_set() or self._stopping.is_set():
            raise SubmitError("server is shutting down", status_code=503)
        if len(prompt_token_ids) == 0:
            raise SubmitError("prompt tokenized to zero tokens", status_code=400)
        if max_new_tokens <= 0:
            raise SubmitError("max_new_tokens must be positive", status_code=400)
        request = GenerationRequest(
            request_id=uuid4().hex, prompt_token_count=len(prompt_token_ids),
            max_new_tokens=max_new_tokens, prompt_token_ids=list(prompt_token_ids),
        )
        handle = RequestHandle(request)
        try:
            self._inbox.put_nowait(handle)
        except Full as error:
            raise SubmitError("submission queue is full", status_code=429) from error
        self._wake.set()
        return handle

    def cancel(self, handle: RequestHandle, *, reason: str = "CLIENT_DISCONNECTED") -> None:
        """Request cancellation without allowing a handler to mutate engine state."""
        if handle.completed.is_set():
            return
        self._cancellations.put((handle.request.request_id, reason))
        self._wake.set()

    # ------------------------------------------------------------------ worker internals
    def _drain_inbox(self) -> None:
        while True:
            try:
                handle = self._inbox.get_nowait()
            except Empty:
                return
            if not self.engine.submit(handle.request):
                # Rejected by the scheduler (queue full / can never fit). The request
                # already carries its terminal reason; the API maps it to a status.
                with self._lock:
                    self._failed_count += 1
                handle._complete(RuntimeError(handle.request.finish_reason or "request rejected"))
                continue
            with self._lock:
                self._active[handle.request.request_id] = handle
            handle._accept()

    def _apply_cancellations(self) -> None:
        while True:
            try:
                request_id, reason = self._cancellations.get_nowait()
            except Empty:
                return
            with self._lock:
                handle = self._active.get(request_id)
            if handle is None or handle.request.done:
                continue  # unknown, or finished between the cancel and now
            self.engine.cancel(request_id, reason=reason)
            with self._lock:
                self._active.pop(request_id, None)
                self._cancelled_count += 1
            handle._complete()

    def _publish_completed(self) -> None:
        with self._lock:
            completed = [
                (request_id, handle) for request_id, handle in self._active.items()
                if handle.request.done
            ]
            for request_id, _ in completed:
                self._active.pop(request_id)
            for _, handle in completed:
                if handle.request.state.name == "FAILED":
                    self._failed_count += 1
                else:
                    self._completed_count += 1
        for _, handle in completed:
            handle._complete()

    def snapshot(self) -> dict[str, int | bool]:
        """Return thread-safe service counters without touching GPU engine state."""
        with self._lock:
            return {
                "active_handles": len(self._active),
                "pending_submissions": self._inbox.qsize(),
                "completed_requests": self._completed_count,
                "cancelled_requests": self._cancelled_count,
                "failed_requests": self._failed_count,
                "worker_failed": self._fatal_error is not None,
                "draining": self._draining.is_set(),
            }

    def _fail_all(self, error: BaseException, *, mark_fatal: bool = True) -> None:
        if mark_fatal:
            self._fatal_error = error
        with self._lock:
            handles = list(self._active.values())
            self._active.clear()
        while True:
            try:
                handles.append(self._inbox.get_nowait())
            except Empty:
                break
        for handle in handles:
            handle._complete(error)

    def _worker(self) -> None:
        try:
            while not self._stopping.is_set():
                self._drain_inbox()
                self._apply_cancellations()
                if self.engine.has_unfinished_requests:
                    self.engine.step()
                    self._publish_completed()
                    continue
                self._wake.wait(0.05)
                self._wake.clear()
        except BaseException as error:  # engine-level failure: make it observable, then stop
            self._fail_all(error)
