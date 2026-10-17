"""Fixed calibration text for INT8 KV key smoothing.

`calibrate_key_smoothing` records per-channel key magnitudes on a few thousand tokens of
ordinary text. The benchmarks calibrate on held-out repository files; a served engine
must not depend on those files existing, so this module carries its own text: a mix of
technical prose, a list, numbers and code, which is what chat traffic to a local engine
looks like. The factor it produces is static per model and is validated against the
held-out calibration by `benchmarks/quantization/int8_kv_teacher_forced.py`.
"""

from __future__ import annotations

CALIBRATION_TEXT = """
A paged key-value cache stores the attention state of every running request in fixed-size
blocks drawn from one shared pool. Each request owns a block table that maps its logical
token positions to physical blocks, so two requests never need contiguous memory and a
request can grow one block at a time while it generates. When the pool is exhausted the
scheduler yields the newest request, releases its blocks, and readmits it later; a request
that has already produced tokens either recomputes its prefix or restores an exact copy of
its pages from host memory. The difference matters: recomputing a long prefix stalls every
other request for the length of that prefill, while a restore costs only the transfer time.

Continuous batching changes which requests share a forward pass from one scheduler
iteration to the next. A step may advance eight decoding requests by one token each and
at the same time prefill a 128-token chunk of a prompt that arrived a moment ago. The
projections, norms and feed-forward layers treat all of those tokens as one batch; only
attention has to know which token belongs to which request and which pages it may read.
Because the weights are read once per step whatever the batch holds, the cost of a decode
step at small batch is almost entirely the time to stream the weights from memory, and a
step that carries a prefill chunk costs a little more than the chunk alone would.

The first token of a response is produced by the prefill. Its latency depends on the
prompt length, the chunk size, and whether anything else is decoding at the same time. An
idle engine should prefill a whole prompt in one forward pass. A busy engine should split
the prompt so that no decoding request waits longer than one chunk between tokens; that is
the trade between time to first token and inter-token latency, and no single constant is
right for both a chat assistant and a batch summarization job.

Quantization reduces the bytes read per step. Eight-bit weights halve the traffic of the
large feed-forward matrices and the vocabulary projection, and four-bit weights halve it
again at the cost of a slower, dequantizing matrix multiply for long prompts. An eight-bit
cache doubles the number of tokens a pool can hold, which removes queueing and preemption
under load, and it changes the generated text slightly: a few tokens per hundred differ
from the sixteen-bit result at positions where two candidates were nearly tied.

Things to check before trusting a benchmark:

1. The two configurations ran interleaved, not one after the other, so clock and thermal
   drift fall on both sides.
2. The workload is the same seed, the same prompts and the same arrival times.
3. Every run drains to zero used pages, with no request left waiting or failed.
4. A difference smaller than the run-to-run spread is reported as unresolved.
5. The first eight generated tokens of each identity prompt match the reference model.

Typical numbers on an 8 GB desktop card: a 0.6 billion parameter model decodes at about
180 tokens per second for one request and about 560 tokens per second for eight; a 1.7
billion parameter model decodes at 70 and 320; a 4 billion parameter model with four-bit
weights decodes at 80 for one request. Prefilling a 2,048-token prompt takes 150 ms, 320 ms
and 1,300 ms respectively. The memory bandwidth is 257 GB/s, the level-two cache is 24 MB,
and the host link transfers about 1.4 GB/s. Peak memory for the largest configuration was
5.40 GB, of which 2.51 GB were weights and 1.92 GB were cache pages.

def plan_prefill(requests, chunk_size, token_budget):
    plans = []
    for request in requests:
        if request.state != "PREFILLING":
            continue
        count = min(request.remaining_prefill_tokens, chunk_size, token_budget)
        if count:
            plans.append((request, count))
            token_budget -= count
        if token_budget == 0:
            break
    return plans

The function above is round-robin: each prefilling request receives at most one chunk per
step, and the budget bounds the total prefill work in that step. Fairness between a long
prompt and a short one therefore depends on the chunk size, not on arrival order alone.

Questions people ask such a system, in no particular order: What is the capital of
Australia, and why is it not Sydney? Summarize the following meeting notes in three
bullet points. Translate "the train leaves at half past seven" into French. Write a Python
function that returns the n-th Fibonacci number without recursion. Explain the difference
between latency and throughput to a product manager. Draft a polite email declining an
invitation. Convert 72 degrees Fahrenheit to Celsius and show the formula. Why does bread
rise, and what happens if the dough is left too long? List five common causes of a slow
database query. What were the main causes of the First World War?

Dear Ms. Okafor, thank you for the invitation to speak at the October workshop. I am
afraid I will be travelling that week and cannot attend, but I would be glad to join a
later session or to send written notes on the topic if that would help. With best wishes,
Daniel.

The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. The
year 1969, the number 3.14159, the address 221B Baker Street, the time 09:45, and the sum
$1,234.56 are all tokens a model must represent precisely. Mixed-case identifiers such as
getUserById, snake_case_names, file paths like /usr/local/bin/python3 and URLs like
https://example.org/docs?page=2 appear constantly in developer traffic.

In summary: measure end to end, keep defaults conservative, record what was rejected and
why, and never let a kernel that wins in isolation change a serving default on its own.
"""


def default_calibration_batches(tokenizer, *, length: int = 1024, count: int = 4) -> list[list[int]]:
    """`count` sequences of `length` tokens cut from the fixed text, repeated if needed."""
    ids = tokenizer(CALIBRATION_TEXT.strip()).input_ids
    if not ids:
        raise ValueError("calibration text produced no tokens")
    repeats = -(-(length * count) // len(ids))
    stream = (ids * repeats)[:length * count]
    return [stream[start:start + length] for start in range(0, length * count, length)]
