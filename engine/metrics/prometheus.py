"""Prometheus text-format metrics for the serving process.

The engine already computes everything an operator needs - per-request queue time, TTFT,
inter-token latency, recompute cost, KV utilization, decode batch size, graph captures
that happened during serving - but only as Python dicts read by benchmarks. This turns
the same numbers into a `/metrics` endpoint so the running server can be watched by the
tooling every other serving stack is watched by.

No client library: the text format is a dozen lines of string building, and a serving
process should not take a dependency to expose four metric types. Histogram buckets are
cumulative `le` buckets as the format requires.

Recording happens once per request, on the worker thread that completes it, so the HTTP
path never touches engine state. `render()` takes the lock and formats.
"""

from __future__ import annotations

from threading import Lock

# Seconds. Chosen around the measured operating points: sub-millisecond steps are not
# interesting, and anything past a minute is a timeout rather than a latency.
_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
_TOKEN_BUCKETS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4096)


class _Histogram:
    def __init__(self, buckets: tuple[float, ...]):
        self.buckets = buckets
        self.counts = [0] * len(buckets)
        self.total = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        self.total += value
        self.count += 1
        for index, edge in enumerate(self.buckets):
            if value <= edge:
                self.counts[index] += 1

    def render(self, name: str, help_text: str) -> list[str]:
        lines = [f"# HELP {name} {help_text}", f"# TYPE {name} histogram"]
        cumulative = 0
        for edge, count in zip(self.buckets, self.counts):
            cumulative += count
            lines.append(f'{name}_bucket{{le="{edge}"}} {cumulative}')
        lines.append(f'{name}_bucket{{le="+Inf"}} {self.count}')
        lines.append(f"{name}_sum {self.total}")
        lines.append(f"{name}_count {self.count}")
        return lines


class ServerMetrics:
    """Counters, gauges and histograms for one serving process."""

    def __init__(self, namespace: str = "inference"):
        self.namespace = namespace
        self._lock = Lock()
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, _Histogram] = {
            "time_to_first_token_seconds": _Histogram(_LATENCY_BUCKETS),
            "inter_token_latency_seconds": _Histogram(_LATENCY_BUCKETS),
            "e2e_request_latency_seconds": _Histogram(_LATENCY_BUCKETS),
            "request_queue_time_seconds": _Histogram(_LATENCY_BUCKETS),
            "request_prompt_tokens": _Histogram(_TOKEN_BUCKETS),
            "request_generation_tokens": _Histogram(_TOKEN_BUCKETS),
        }
        self._help = {
            "time_to_first_token_seconds": "Time from arrival to the first generated token.",
            "inter_token_latency_seconds": "Mean gap between generated tokens, per request.",
            "e2e_request_latency_seconds": "Arrival to final token, including queueing and stalls.",
            "request_queue_time_seconds": "Time spent waiting for admission, including after preemption.",
            "request_prompt_tokens": "Prompt length per request.",
            "request_generation_tokens": "Generated tokens per request.",
        }

    # ------------------------------------------------------------------ recording
    def increment(self, name: str, value: float = 1.0, **labels: str) -> None:
        key = _key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + value

    def set_gauge(self, name: str, value: float, **labels: str) -> None:
        key = _key(name, labels)
        with self._lock:
            self._gauges[key] = float(value)

    def observe_request(self, request, outcome: str) -> None:
        """Record one finished request. Called on the worker thread."""
        report = request.latency_report()
        with self._lock:
            self._counters["requests_total" + _labels({"outcome": outcome})] = (
                self._counters.get("requests_total" + _labels({"outcome": outcome}), 0.0) + 1
            )
            self._counters["prompt_tokens_total"] = (
                self._counters.get("prompt_tokens_total", 0.0) + request.prompt_token_count
            )
            generated = len(request.output_token_ids)
            self._counters["generation_tokens_total"] = (
                self._counters.get("generation_tokens_total", 0.0) + generated
            )
            self._counters["preemptions_total"] = (
                self._counters.get("preemptions_total", 0.0) + request.preempted_count
            )
            self._counters["recomputed_tokens_total"] = (
                self._counters.get("recomputed_tokens_total", 0.0)
                + request.recomputed_token_count
            )
            self._histograms["request_prompt_tokens"].observe(request.prompt_token_count)
            self._histograms["request_generation_tokens"].observe(generated)
            for metric, key in (
                ("time_to_first_token_seconds", "ttft_ms"),
                ("inter_token_latency_seconds", "mean_itl_ms"),
                ("request_queue_time_seconds", "total_queue_ms"),
            ):
                value = report.get(key)
                if value:
                    self._histograms[metric].observe(float(value) / 1000.0)
            end_to_end = (report.get("total_queue_ms") or 0.0) + (report.get("generation_ms") or 0.0)
            if end_to_end:
                self._histograms["e2e_request_latency_seconds"].observe(end_to_end / 1000.0)

    def observe_engine(self, stats: dict) -> None:
        """Mirror the engine's snapshot into gauges. Cheap; safe to call every publish."""
        mapping = {
            "requests_running": "active_requests",
            "requests_waiting": "waiting_requests",
            "decode_batch_size": "decode_batch",
            "decode_mean_context_tokens": "decode_mean_context",
            "kv_cache_utilization": "kv_utilization",
            "kv_blocks_used": "kv_blocks_used",
            "kv_blocks_total": "kv_blocks_total",
            "prefix_cache_blocks": "prefix_cache_blocks",
            "graph_captures_in_service": "lazy_graph_captures",
        }
        with self._lock:
            for metric, key in mapping.items():
                if key in stats and stats[key] is not None:
                    self._gauges[metric] = float(stats[key])

    # ------------------------------------------------------------------ output
    def render(self) -> str:
        with self._lock:
            lines: list[str] = []
            for key in sorted(self._counters):
                name, _, labels = key.partition("{")
                lines.append(f"# TYPE {self.namespace}_{name} counter")
                suffix = "{" + labels if labels else ""
                lines.append(f"{self.namespace}_{name}{suffix} {self._counters[key]}")
            for key in sorted(self._gauges):
                name, _, labels = key.partition("{")
                lines.append(f"# TYPE {self.namespace}_{name} gauge")
                suffix = "{" + labels if labels else ""
                lines.append(f"{self.namespace}_{name}{suffix} {self._gauges[key]}")
            for name, histogram in self._histograms.items():
                lines.extend(histogram.render(
                    f"{self.namespace}_{name}", self._help.get(name, name),
                ))
        return "\n".join(lines) + "\n"


def _labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{key}="{value}"' for key, value in sorted(labels.items()))
    return "{" + inner + "}"


def _key(name: str, labels: dict[str, str]) -> str:
    return name + _labels(labels)
