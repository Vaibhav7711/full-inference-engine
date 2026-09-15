"""Thread-owned bridge from async HTTP handlers to continuous GPU batching."""

from __future__ import annotations

from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Sequence
from uuid import uuid4

from engine.runtime import GenerationRequest


@dataclass
class RequestHandle:
    """A caller-owned view of one scheduler request."""

    request: GenerationRequest
    completed: Event = field(default_factory=Event)
    error: BaseException | None = None


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
        self._thread: Thread | None = None
        self._fatal_error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = Thread(target=self._worker, name="continuous-batching-gpu", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def submit(self, prompt_token_ids: Sequence[int], max_new_tokens: int) -> RequestHandle:
        if self._fatal_error is not None:
            raise RuntimeError("continuous batching worker is unavailable") from self._fatal_error
        request = GenerationRequest(
            request_id=uuid4().hex, prompt_token_count=len(prompt_token_ids),
            max_new_tokens=max_new_tokens, prompt_token_ids=list(prompt_token_ids),
        )
        handle = RequestHandle(request)
        try:
            self._inbox.put_nowait(handle)
        except Full as error:
            raise RuntimeError("submission queue is full") from error
        self._wake.set()
        return handle

    def cancel(self, handle: RequestHandle, *, reason: str = "CLIENT_DISCONNECTED") -> None:
        """Request cancellation without allowing a handler to mutate engine state."""
        if handle.completed.is_set():
            return
        self._cancellations.put((handle.request.request_id, reason))
        self._wake.set()

    def _drain_inbox(self) -> None:
        while True:
            try:
                handle = self._inbox.get_nowait()
            except Empty:
                return
            if not self.engine.submit(handle.request):
                handle.error = RuntimeError(handle.request.finish_reason or "request rejected")
                handle.completed.set()
                continue
            with self._lock:
                self._active[handle.request.request_id] = handle

    def _apply_cancellations(self) -> None:
        while True:
            try:
                request_id, reason = self._cancellations.get_nowait()
            except Empty:
                return
            with self._lock:
                handle = self._active.pop(request_id, None)
            if handle is None or handle.request.done:
                continue
            self.engine.cancel(request_id, reason=reason)
            handle.completed.set()

    def _publish_completed(self) -> None:
        with self._lock:
            completed = [request_id for request_id, handle in self._active.items() if handle.request.done]
            for request_id in completed:
                handle = self._active.pop(request_id)
                handle.completed.set()

    def _fail_all(self, error: BaseException) -> None:
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
            handle.error = error
            handle.completed.set()

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
        except BaseException as error:  # make every waiting HTTP request observable
            self._fail_all(error)
