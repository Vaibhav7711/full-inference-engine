# Kernel study guide: every GPU path in `full-inference-engine`

This is the code-grounded study manual for the kernels in this repository. It describes
the current tree, including the newer FP8 KV and W4A16 work that is newer than parts of the
README. It does not treat every implementation as a production default: each section labels
the path as **live default**, **live optional**, **experimental/unreleased**, or **historical**.

The document is meant to be read in order. The early chapters establish the tensor layouts,
Triton vocabulary, and online-softmax invariant that the later attention kernels reuse.

---

## 1. Kernel map and recommended study order

| Order | File | Role | Status |
|---:|---|---|---|
| 1 | `engine/kernels/rmsnorm.py` | Fused RMS normalization | Live default when enabled |
| 2 | `engine/kernels/rope.py` | Combined query/key rotary embedding | Live default when enabled |
| 3 | `engine/kernels/swiglu.py` | Fused `SiLU(gate) * up` | Live default when enabled |
| 4 | `engine/kernels/kv_write.py` | FP16/BF16 decode and prefill writes into paged KV | Live |
| 5 | `engine/kernels/triton_attention.py` | K1: contiguous tiled attention | Historical correctness rung |
| 6 | `engine/kernels/paged_attention_kernel.py` | K2: one-sequence paged attention | Historical correctness rung |
| 7 | `engine/kernels/paged_kernel_attention.py` | K3: K2 wired into a real model | Historical integration rung |
| 8 | `engine/kernels/paged_decode_batched.py` | K4: multi-request paged decode | Live T4/Ada default |
| 9 | `engine/kernels/paged_decode_config.py` | Decode tile and split policy | Live |
| 10 | `engine/kernels/paged_decode_gqa.py` | Share one K/V tile across two query heads | Optional negative result |
| 11 | `engine/kernels/paged_decode_split_k.py` | Flash-Decoding-style context splitting | Optional |
| 12 | `engine/kernels/paged_prefill.py` | One program per prefill query token | Live fallback/reference |
| 13 | `engine/kernels/sdpa_prefill.py` | Gather pages and call PyTorch SDPA | Live default on sm_75 |
| 14 | `engine/kernels/tiled_paged_prefill.py` | FlashAttention-style tiled paged prefill | Live default on measured sm_89; refused on sm_75 |
| 15 | `engine/kernels/flash_paged.py` | FlashAttention adapter for paged/dense paths | Optional, eager-only on tested build |
| 16 | `engine/kernels/int8_paged_kv.py` | INT8 KV writers and dequantizing attention | Live optional/off by default |
| 17 | `engine/kernels/fp8_paged_kv.py` | FP8 E4M3 KV writers and attention | Experimental/unreleased, sm_89+ |
| 18 | `engine/kernels/w4a16_linear.py` | Grouped INT4-weight/FP16-activation linear | Serving-integrated, active validation |
| 19 | `engine/kernels/w8a16_linear.py` | Per-channel INT8-weight/FP16-activation linear | Experimental negative result |
| 20 | `engine/kernels/w8a8_linear.py` | Dynamic INT8 activation and CUDA INT8 GEMM | Experimental negative result |

Supporting files are part of the kernel contract even though they do not themselves launch
Triton programs:

- `engine/kernels/device.py`: device profile and prefill tile selection.
- `engine/kernels/fp8_format.py`: CPU-testable FP8 storage contract.
- `engine/kernels/w4a16_format.py`: CPU-testable INT4 packing contract.
- `engine/backends/registry.py`, `builtin.py`, `policy.py`: eligibility and selection.
- `engine/batching/continuous_batching.py`: model hooks and actual launch flow.

---

## 2. The complete engine flow around the kernels

### 2.1 Decode flow

For `N` active requests, the engine stages one pending token per request:

```text
input_ids      [N, 1]
position_ids   [N, 1]
seq_lens       [N]              committed length before this token
block_tables   [N, max_blocks]  logical page -> physical page
```

The Hugging Face decoder performs the dense projections. In each layer:

```text
hidden state
    │
    ├─ RMSNorm                         Triton fused kernel
    ├─ Q/K/V linear projections        PyTorch/cuBLAS or W4A16 path
    ├─ Q/K normalization               Triton RMSNorm
    ├─ RoPE(Q, K)                      Triton combined Q/K kernel
    ├─ write K,V at logical position   paged KV-write kernel
    ├─ attention over cached history   selected paged decode kernel
    ├─ output projection
    ├─ residual + RMSNorm
    └─ gate/up → SwiGLU → down          Triton SwiGLU between linears
```

The attention hook is `batched_decode_attention_forward()` in
`engine/batching/continuous_batching.py`. Its crucial order is:

```python
write_decode_kv(key, value, key_pool, value_pool, block_tables, seq_lens)
out = decode_backend.run(
    query, key_pool, value_pool, block_tables, seq_lens,
    length_offset=1,
)
```

`seq_lens` is still the committed length before the step. The writer stores at exactly
that position. Attention uses `length_offset=1`, so it sees the newly stored token too.
Only after the full forward succeeds does the CPU-side block manager commit the new length.
That ordering prevents a failed forward from leaving logical state ahead of valid logits.

### 2.2 Chunked prefill flow

For `B` prompt chunks of padded width `Q`:

```text
query/key/value [B, heads, Q, D]
start_positions [B]      absolute first position of each chunk
chunk_lens      [B]      real tokens in each padded row
block_tables    [B, max_blocks]
```

Every layer first writes the rotated K/V chunk into the page pool, then the selected prefill
backend computes causal attention over the previously committed prefix plus the current chunk.

```python
write_prefill_kv_batched(..., start_positions)
out = prefill_backend.run(
    query, key_pool, value_pool, block_tables,
    start_positions, chunk_lens,
)
```

### 2.3 Fused decode + prefill flow

The fused step packs decode tokens followed by flattened prefill rows into one model input:

```text
[ decode token 0 ... decode token N-1 | chunk row 0 | chunk row 1 ... ]
```

Dense layers run once over this combined `M` dimension, so weights are read once. Only the
attention hook separates the tokens again because decode and prefill require different cache
metadata and attention algorithms. It runs the decode kernel on `[N,H,1,D]`, the prefill
kernel on `[B,H,Q,D]`, and concatenates the outputs back into model order.

### 2.4 CUDA graphs do not replace kernels

A CUDA graph records the sequence of these kernel launches and replays it. The kernels remain
the same. Persistent device buffers keep addresses fixed while input IDs, lengths, and block
tables change contents. Graph keys include the row bucket, prefill context bucket, attention
kind, and decode tile regime when those facts change captured execution.

---

## 3. Common tensor layouts and Triton vocabulary

The recurring symbols are:

```text
B   batch/prefill rows
S   decode sequences
Hq  query heads
Hkv key/value heads
Q   query/chunk length
T   logical KV history length
D   head dimension
P   number of physical pages
BS  tokens per physical page
```

The authoritative page layout is:

```text
key_pool[layer]   [P, BS, Hkv, D]
value_pool[layer] [P, BS, Hkv, D]
```

For logical token position `t`:

```text
logical_page = t // BS
slot         = t % BS
physical     = block_table[row, logical_page]
address      = pool[physical, slot, kv_head, dimension]
```

The general strided address equation is:

```text
ptr + i0*stride0 + i1*stride1 + ... + in*striden
```

Triton concepts used throughout:

- `tl.program_id(axis)`: identifies the independent work item on one grid axis.
- `tl.arange(0, BLOCK)`: produces a vector of lane offsets inside the program.
- `tl.load(..., mask=..., other=...)`: safe vector load; invalid lanes get `other`.
- `tl.store(..., mask=...)`: invalid lanes perform no write.
- `tl.constexpr`: compile-time specialization, used for dimensions and tile sizes.
- `num_warps`: CUDA warps assigned to one Triton program.
- `tl.dot`: tiled matrix product that may reach tensor cores on supported shapes/devices.

### 3.1 Online softmax invariant

The attention kernels do not materialize the full score matrix. For each processed key tile
they maintain:

```text
m   running maximum score
l   running sum of exp(score - m)
acc running sum of exp(score - m) * value
```

When a new tile has maximum `m_tile`:

```text
m_new = max(m, m_tile)
alpha = exp(m - m_new)
p     = exp(scores_tile - m_new)

acc = acc * alpha + sum(p * V_tile)
l   = l   * alpha + sum(p)
m   = m_new
```

Final output is `acc / l`. Rescaling the old state by `alpha` puts both old and new
contributions on the same numerical scale. This is the core invariant in K1, K2, K4,
per-token prefill, GQA-shared decode, Split-K, INT8 KV, and FP8 KV.

---

## 4. Fused RMSNorm

**File:** `engine/kernels/rmsnorm.py`  
**Status:** live when `triton_rmsnorm=True`.

### Contract and math

For each final-dimension row of width `W`:

```text
variance = (1/W) Σ x_i²
inverse_rms = 1 / sqrt(variance + epsilon)
y_i = cast(x_i * inverse_rms, input_dtype) * weight_i
```

The cast before the weight multiply is deliberate: it matches Qwen’s reference operation
ordering closely enough for the token-identity gate.

### Launch ownership

```text
grid = (rows,)
one program = one complete normalization row
BLOCK = next_power_of_2(W)
```

Core code:

```python
row = tl.program_id(0)
columns = tl.arange(0, BLOCK)
mask = columns < WIDTH
values = tl.load(input_ptr + row * input_row_stride + columns,
                 mask=mask, other=0.0).to(tl.float32)
variance = tl.sum(values * values, axis=0) / WIDTH
inverse_rms = tl.rsqrt(variance + epsilon)
normalized = (values * inverse_rms).to(input_ptr.dtype.element_ty)
weights = tl.load(weight_ptr + columns, mask=mask, other=0.0)
tl.store(output_ptr + row * output_row_stride + columns,
         normalized * weights, mask=mask)
```

### Why it exists

An unfused expression can launch separate square, reduction, epsilon-add, reciprocal-square-
root, normalize, and weight-multiply operations. The Triton path reads the row, performs the
reduction and normalization, and writes once.

### Correctness hazards

- Accumulating `x²` in FP16 loses accuracy; the kernel promotes to FP32.
- Padding lanes must be zero for the sum-of-squares reduction.
- `W > 65536` is refused because one-program-per-row becomes unsupported.
- The installer stores the original unbound forward function to avoid a module retention
  cycle and supports an exact uninstall.

### Tradeoffs

- Excellent for moderate row widths reused thousands of times in a generation.
- One program owns one row, so extremely wide rows raise register/shared reduction pressure.
- This is an elementwise/reduction optimization, not the main decode bandwidth lever.

---

## 5. Combined Q/K RoPE

**File:** `engine/kernels/rope.py`  
**Status:** live when `triton_rope=True`.

### Contract

```text
query [B, Hq, S, D]
key   [B, Hkv, S, D]
cos   [1 or B, S, D]
sin   [1 or B, S, D]
```

Qwen’s half-rotation pairs the first and second halves of the head vector:

```text
rotated_dims = d + D/2  if d < D/2
               d - D/2  otherwise
sign          = -1      in first half
                +1      in second half

rope(x)_d = x_d*cos_d + sign*x_rotated_dims*sin_d
```

### Launch ownership

```text
grid = (B, S, max(Hq, Hkv))
one program = one batch/token/head vector
```

The single program loads `cos` and `sin` once, writes Q if `head < Hq`, and writes K if
`head < Hkv`. This combines the two rotations despite GQA having different head counts.

### Important code idea

```python
rotated_dims = tl.where(dims < half, dims + half, dims - half)
signs = tl.where(dims < half, -1.0, 1.0)
q_out = q * cos + q_rotated * signs * sin
k_out = k * cos + k_rotated * signs * sin
```

### Engine significance

K is written into the KV cache **after** RoPE. Historical keys are therefore already in
rotated space, and future queries can attend to them without rerunning RoPE on history.

### Correctness hazards and tradeoffs

- `D` must be even and at most 256.
- A `[1,S,D]` table is expanded as a zero-stride batch view, avoiding a copy.
- The patch is module-global because Transformers looks up `apply_rotary_pos_emb` in the
  model module. `stock_rope` temporarily restores the original for an independent oracle.
- It saves launches and intermediate traffic; it does not alter the attention complexity.

---

## 6. Fused SwiGLU

**File:** `engine/kernels/swiglu.py`  
**Status:** fused elementwise path live; gate/up projection fusion optional.

### Math

```text
SiLU(g) = g * sigmoid(g)
output  = SiLU(gate) * up
```

### Launch ownership

```text
inputs viewed as [rows, columns]
grid = (rows, ceil(columns / 512))
one program = a 512-column segment of one row
```

Core code:

```python
gate = tl.load(...).to(tl.float32)
up = tl.load(...).to(tl.float32)
tl.store(..., gate * tl.sigmoid(gate) * up, mask=mask)
```

The FP32 promotion is both a Triton requirement for `sigmoid` on the tested version and a
numerically stable match for PyTorch SiLU before casting the stored output.

### Strided-half support

`FusedGateUpProjection` can concatenate gate/up weights and return two strided views of one
linear output. `_as_rows()` preserves those views when their final dimension is contiguous,
so the activation kernel does not pay two full copies merely to make halves contiguous.

### Measured tradeoff

- Elementwise SwiGLU fusion is kept.
- Fusing the **linear projections** was noisy around parity and is not the defensible default.
- Dense GEMM still dominates MLP work; this kernel removes surrounding pointwise overhead.

---

## 7. Paged KV writers

**File:** `engine/kernels/kv_write.py`  
**Status:** live authoritative FP16/BF16 writers.

### 7.1 Decode writer

Input and destination layouts:

```text
key/value       [N, Hkv, 1, D]
key/value pool  [P, BS, Hkv, D]
block tables    [N, max_blocks]
seq_lens        [N]
```

Launch:

```text
grid = (N, Hkv)
one program = one request row and one KV head, all D elements
```

Address translation:

```python
position = tl.load(seq_lens_ptr + batch)
logical_block = position // BLOCK_SIZE
block_offset = position % BLOCK_SIZE
physical_block = tl.load(block_tables_ptr + batch*stride_bt_b + logical_block)

destination = (
    pool_ptr
    + physical_block * stride_page
    + block_offset * stride_slot
    + head * stride_head
    + dims * stride_dim
)
```

Host-side lifecycle validation ensures the destination page exists. The kernel adds a
device-side backstop: it verifies table bounds and that `0 <= physical_block < P`, masking
the store otherwise. That prevents corrupted metadata from becoming a cross-request write.

### 7.2 Prefill writer

Input:

```text
key/value       [B, Hkv, Q, D]
start_positions [B]
chunk_lens      [B]
```

Launch:

```text
grid = (B, padded_Q, Hkv)
one program = one row/token/head vector
position = start_positions[row] + token
```

`token < chunk_lens[row]` masks padded rows. Because every token independently translates
its absolute position, a chunk crossing any number of page boundaries needs no CPU special
case.

### Performance model

These are pure scatter/write kernels. They move `2 * Hkv * D * element_size` bytes per token
per layer plus small table metadata. They avoid much larger costs: cache concatenation,
per-request allocation, and gathering old history before attention.

---

## 8. Attention ladder: K1, K2, K3, K4

The historical ladder is valuable because each rung changed one concept and kept a clear
correctness oracle.

```text
K1 contiguous tiled attention == PyTorch SDPA
       ↓ change only K/V addressing
K2 single-sequence paged attention == K1
       ↓ wire into real model/cache
K3 real token generation == stock model
       ↓ add independent rows and GQA mapping
K4 continuously batched paged decode == per-row reference
```

### 8.1 K1: contiguous tiled attention

**File:** `engine/kernels/triton_attention.py`  
**Status:** historical/reference.

Contract:

```text
Q [B,H,M,D]
K [B,H,N,D]
V [B,H,N,D]
```

`B` and `H` flatten into `Z`; grid is `(ceil(M/BLOCK_M), B*H)`. Each program owns a query
tile and streams K/V in `BLOCK_N` tiles. `tl.dot(Q, Kᵀ)` and `tl.dot(P, V)` implement the
two matrix products, with online softmax between them.

The explicit `input_precision="ieee"` on FP32 avoids Triton selecting TF32 on Ada and
violating a tight numerical test. FP16/BF16 inputs still use the appropriate MMA path.

### 8.2 K2: single-sequence paged attention

**File:** `engine/kernels/paged_attention_kernel.py`  
**Status:** historical/reference.

K2 keeps K1’s tiling and online softmax. The only conceptual change is:

```python
logical_block = logical_position // BLOCK_SIZE
offset = logical_position % BLOCK_SIZE
physical = block_table[logical_block]
```

The query remains `[H,M,D]`; pages are `[P,BS,H,D]`; the grid is
`(ceil(M/BLOCK_M), H)`. The helper `scatter_to_pages()` exists only to construct a paged
test fixture from contiguous K/V and prove K2 equals K1.

### 8.3 K3: live single-request model integration

**File:** `engine/kernels/paged_kernel_attention.py`  
**Status:** historical integration rung, not current serving path.

K3 installed K2 as the Hugging Face attention function. Its early cache allocated pages
contiguously, so the block table was the identity. GQA pages were explicitly repeated to
query-head count, which is simple but physically duplicates a view/materialization concern.
It proved that the model could generate correct tokens while attention read physical pages.

Its process-global active-cache handle also revealed a limitation later formalized in the
current `_BATCH_CTX`/`_PREFILL_CTX` design: one process cannot safely overlap independent
engine forwards through a single global attention callback context.

### 8.4 K4: multi-sequence paged decode

**File:** `engine/kernels/paged_decode_batched.py`  
**Status:** live measured default on sm_75 and sm_89.

Contract:

```text
Q             [S, Hq, 1, D]
K/V pages     [P, BS, Hkv, D]
block_tables  [S, max_blocks]
seq_lens      [S]
output        [S, Hq, 1, D]
```

Launch:

```text
grid = (S, Hq)
one program = one sequence and one query head
```

GQA mapping is arithmetic, not a repeated page tensor:

```python
n_rep = Hq // Hkv
kv_head = query_head // n_rep
```

Each program loads one query vector and streams only its request’s logical history through
that row’s block table. Since the decode query is the newest token, it sees the whole cache;
there is no per-score causal mask beyond `position < seq_len`.

Core loop:

```python
for start_n in range(0, seq_len, BLOCK_N):
    positions = start_n + tl.arange(0, BLOCK_N)
    physical = block_table[positions // BLOCK_SIZE]
    slots = positions % BLOCK_SIZE
    K = key_pages[physical, slots, kv_head, :]
    scores = sum(query * K, axis=D) * scale
    # online softmax update
    V = value_pages[physical, slots, kv_head, :]
    acc += probabilities * V
```

Unlike K1/K2, K4 uses elementwise multiply/reduce for a single query rather than a query
matrix `tl.dot`: `M=1` is decode-shaped and a matrix tile would mostly be empty.

### Decode tile regime

`paged_decode_config.py` currently selects:

```text
maximum context < 128  → BLOCK_N=64,  4 warps
maximum context ≥ 128  → BLOCK_N=128, 4 warps
```

This was measured on a T4; it is not a universal law. Larger tiles reduce loop overhead and
improved measured 256–2048-token cases, but consume more registers and can reduce occupancy.

### Performance limit

At batch ≥ 4, the project’s measurements place the kernel near the relevant DRAM roofline.
Once memory bandwidth is saturated, further arithmetic rearrangement cannot yield large
end-to-end improvements; reducing bytes (KV quantization) or improving another component is
the plausible lever.

---

## 9. GQA-shared decode

**File:** `engine/kernels/paged_decode_gqa.py`  
**Status:** optional negative result; requires GQA group exactly 2.

Qwen3’s 16 query heads and 8 KV heads mean two query heads consume the same K/V head. The
per-head kernel appears to load the same K/V tile twice. This variant launches one program
per `(sequence, KV head)`, loads K and V once, and keeps two separate online-softmax states:

```text
K/V tile once
   ├─ q0 → m0, l0, acc0
   └─ q1 → m1, l1, acc1
```

An initial rank-3 implementation created `[REP,BLOCK_N,D]` intermediates and was 1.5–2.8x
slower. The current version explicitly unrolls two rank-2 paths, isolating the true shared
load idea.

Measured result was approximately 0.95–1.06x—noise. The reason is subtle: two source-code
loads do not imply two DRAM loads. L2 already served the second query head’s K/V read, while
the GQA kernel halved the grid width and added two simultaneous softmax accumulator sets.

Interview lesson: always distinguish logical traffic from actual DRAM traffic.

---

## 10. Split-K paged decode

**File:** `engine/kernels/paged_decode_split_k.py`  
**Status:** live optional, mainly small batch/long context.

At batch 1 with 16 query heads, the per-head grid has only 16 programs—too little parallelism
for a 40-SM T4. Split-K divides the context across `K` programs per row/head:

```text
grid = (rows, query_heads, splits)
```

Each split writes partial state:

```text
partial_acc [rows, heads, splits, D] FP32
partial_m   [rows, heads, splits]    FP32
partial_l   [rows, heads, splits]    FP32
```

The merge kernel computes:

```text
global_m = max(m_k)
r_k = exp(m_k - global_m)
global_l = Σ l_k * r_k
global_acc = Σ acc_k * r_k
output = global_acc / global_l
```

`choose_splits()` tries to create at least two programs per SM, limits each slice to at
least one full `BLOCK_N` tile, caps splits, and rounds down to a power of two. If it chooses
one, the wrapper calls K4 directly and avoids workspace plus merge launch.

Tradeoff: more occupancy versus a second kernel, FP32 workspace, and merge traffic. It only
pays when the original grid is too narrow.

---

## 11. Per-token paged prefill

**File:** `engine/kernels/paged_prefill.py`  
**Status:** live fallback/reference; native structure mirrored by INT8/FP8.

Contract:

```text
query          [B, Hq, Q, D]
K/V pages      [P, BS, Hkv, D]
start_positions[B]
chunk_lens     [B]
```

Launch:

```text
grid = (B, Hq, Q)
one program = one query token and one query head
```

For query index `j`, the absolute position is `start[row] + j`; the visible KV length is
`start[row] + j + 1`. Therefore the causal rule is encoded by the loop bound itself rather
than a score mask over future keys.

The algorithm is K4 repeated for every query token. That is simple and reads pages in place,
but query tokens in the same chunk repeatedly stream overlapping prefixes. For a 128-token
chunk and 16 query heads, that is 2,048 programs repeatedly reading much of the same K/V.
On T4, gathered SDPA is substantially faster. The path remains useful as the direct paged
reference and as the structural basis for scaled INT8/FP8 pools.

---

## 12. Gathered-page SDPA prefill

**File:** `engine/kernels/sdpa_prefill.py`  
**Status:** live sm_75 default. This is composed PyTorch GPU work, not a custom Triton kernel.

The path performs four stages:

1. Turn each block table’s first `ceil(total_len/BS)` page IDs into one index vector.
2. `index_select` page tensors and reshape them into dense `[B,Hkv,T,D]` K/V.
3. Build a boolean `[B,1,Q,T]` causal mask.
4. Call `F.scaled_dot_product_attention`.

The mask treats padded query rows specially: each is allowed to see key zero, so softmax has
one finite entry and cannot produce NaN. Its output is ignored.

### GQA folding

On the tested T4 stack, `enable_gqa=True` selected an unfavorable math path. Instead, query
heads sharing a KV head are folded into the query-length axis:

```text
Q [B, Hq, Q, D]
  → [B, Hkv, repeat*Q, D]
K/V remain [B, Hkv, T, D]
```

After SDPA, output reshapes back to `[B,Hq,Q,D]`. This preserves the mapping
`kv_head = query_head // repeat` without materializing repeated K/V heads.

### Per-step metadata cache

Page indices and the causal mask depend on batch metadata, not layer number. `_PrefillContext`
caches them so 28 layers do not rebuild about 15 small operations each. This is an example of
an optimization outside the mathematical kernel that matters because the forward is
launch-sensitive.

### Tradeoff

Gathering copies the prefix every layer, which sounds wasteful. Yet prefill has many queries
and enough matrix work for the optimized SDPA implementation to win. Decode has only one
query, so doing this every token would be a bad trade.

---

## 13. Tiled paged prefill

**File:** `engine/kernels/tiled_paged_prefill.py`  
**Status:** refused on sm_75; measured default on sm_89; also used as an experimental
speculative-verification backend.

This is a FlashAttention-style kernel adapted to page gathers:

```text
grid = (B, Hq, ceil(Q/BLOCK_M))
one program = BLOCK_M query rows for one batch row/head
```

It loads a `[BLOCK_M,D]` Q tile, streams `[BLOCK_N,D]` K/V tiles through the block table,
uses `tl.dot(Q,Kᵀ)` and `tl.dot(P,V)`, and carries vector online-softmax state for all
`BLOCK_M` queries.

### Causal bound

For one query tile, no row can see past the tile’s last real query position:

```text
tile_last = min(tile_start + BLOCK_M, chunk_len)
kv_end = chunk_start + tile_last
```

This avoids visiting K/V tiles that are future to every query row in the tile. A per-element
visibility mask still enforces `key_position <= query_absolute_position`.

### Finite masked sentinel

The kernel uses `-1e30` instead of `-inf`. A fully padded row would otherwise evaluate
`(-inf) - (-inf)` during online-softmax rescaling and create NaN. It then explicitly forces
masked probabilities to zero.

### Paged gather and transpose tension

The pool layout makes `[BLOCK_N,D]` K loads coalesced, but `QKᵀ` wants `[D,BLOCK_N]` as the
right operand. The kernel exposes two measurable variants:

- contiguous load followed by `tl.trans(keys)`;
- transposed/strided load directly.

### Architecture result

On sm_75, `tl.dot` did not lower to the desired tensor-core instruction, while the kernel
paid high register/shared-memory costs and spills. It was about 3x slower and is explicitly
refused. On measured sm_89, it reached tensor cores and beat SDPA for the tested long profile,
so policy selects it there. Algorithm quality and compiler/hardware realization must be
separated.

### Resource fallback

If the compiler reports shared-memory exhaustion and the caller did not force a stage count,
the wrapper retries with shallower pipelining. It does not silently change an explicit choice.

---

## 14. FlashAttention adapters

**File:** `engine/kernels/flash_paged.py`  
**Status:** optional; these functions adapt tensors rather than implement attention.

### Direct paged decode

`flash_attn_with_kvcache` accepts `[P,page_size,Hkv,D]`, a block table, and cache lengths.
The adapter transposes query from `[S,H,1,D]` to `[S,1,H,D]`, adds `length_offset`, calls
FlashAttention, and transposes back. It handles GQA and internal splits natively.

### Direct paged prefill

With `causal=True`, FlashAttention bottom-right aligns the query chunk with the end of its
cache. The API has no per-row query-length vector, so padded rows with different real lengths
must be grouped by `chunk_len`; otherwise causal alignment silently changes outputs.

### Dense Flash prefill

For the normal 16-token page layout, direct Flash is unavailable because the tested paged
API requires page size divisible by 256. `flash_dense_prefill` gathers pages like SDPA and
then calls `flash_attn_func` group-by-group. In the measured 16-token-page case, gather and
materialization dominated the better compute kernel and the path lost.

### Eligibility and end-to-end tradeoff

- Direct paged API: page size divisible by 256, head dim divisible by 8 and at most 256,
  FP16/BF16, compatible installed wheel, sm_80+.
- The tested FA2 K/V-cache path performs workspace setup that is not graph-capture safe.
- An isolated kernel win can lose end-to-end if selecting it forces the whole model forward
  out of CUDA graph replay. That is why the backend is marked `graph_safe=False`.

---

## 15. INT8 paged KV

**File:** `engine/kernels/int8_paged_kv.py`  
**Status:** end-to-end optional, disabled by default.

### Storage contract

For each token and KV head independently:

```text
scale = max(abs(vector)) / 127, floored at 1e-8
q = round(vector / scale), clamped to [-127,127], stored int8
dequantized ≈ q * scale
```

Pools:

```text
K/V data   [P, BS, Hkv, D] int8
K/V scales [P, BS, Hkv]    fp16
```

### Writers

Decode grid is `(N,Hkv)`; prefill grid is `(B,Q,Hkv)`. Each program reduces its vector to
find `max(abs(x))`, explicitly implements nearest rounding because a raw Triton float-to-int
cast truncates, stores INT8 data and one scale.

### Attention

INT8 decode and prefill mirror the FP16 per-head/per-token algorithms. On every K/V tile:

```python
keys = load(int8_keys).to(fp32) * key_scales[:, None]
values = load(int8_values).to(fp32) * value_scales[:, None]
```

Dequantization is fused into consumption; the engine never materializes a full FP16 cache.

### Tradeoffs

- Approximately halves KV data bytes; scale storage makes the total reduction slightly less
  than exactly 50%.
- Can raise the long-context/high-concurrency ceiling and improve a bandwidth-bound attention
  operating point.
- Adds max-reduction, quantization, scale traffic, and dequantization arithmetic.
- Small batches/short contexts may be slower; numerical approximation can flip near-tied
  logits; a recompute path exposed drift on Ada. Therefore it is not the global default.

---

## 16. FP8 E4M3 paged KV

**Files:** `engine/kernels/fp8_format.py`, `fp8_paged_kv.py`  
**Status:** integrated experimental/unreleased path, gated to sm_89+.

### Storage contract

E4M3’s largest finite value is 448 and it has no infinity. For each vector:

```text
scale = max(abs(vector)) / 448, floored at 1e-8
q = clamp(vector / scale, -448, 448).to(float8_e4m3fn)
dequantized = q.float() * scale
```

The clamp is correctness-critical: an overflowing E4M3 cast produces NaN rather than
infinity. As with INT8, scale pools are `[P,BS,Hkv]` FP16.

### Kernel family

`fp8_paged_kv.py` contains four Triton kernels matching the INT8 structure:

- decode FP8 writer;
- prefill FP8 writer;
- FP8-dequantizing paged decode;
- FP8-dequantizing per-token paged prefill.

The reference format module is intentionally Triton-free, allowing CPU tests to pin the
quantization contract before first GPU launch.

### Eligibility and tradeoff

The engine refuses FP8 below sm_89 with the device in the reason. This path uses FP8 as a
storage format; attention still converts values for computation. It targets the same memory
capacity/bandwidth problem as INT8 with different quantization behavior and hardware support.
Treat performance claims as provisional until the repository’s stated GPU gate completes.

---

## 17. W4A16 weight format and linear kernel

**Files:** `engine/kernels/w4a16_format.py`, `w4a16_linear.py`,
`engine/quantization/w4a16.py`  
**Status:** model/loader/server integrated, active unreleased validation.

### Why weight quantization is different from KV quantization

At decode, the engine reads nearly every model weight to generate the batch’s next tokens.
Once attention is near its roofline and launch overhead is reduced, fewer **weight bytes** are
the largest remaining lever. W4 stores two weights per byte versus two bytes per FP16 weight.

### Format

Weights are symmetric 4-bit values in `[-7,7]`, stored offset by 8 in nibbles. One FP16 scale
is stored per output channel and 128-input group by default.

Packing within a group is split-half:

```text
byte j low nibble  = quantized weight for input j
byte j high nibble = quantized weight for input j + group_size/2
```

This layout lets one `[GROUP/2, BLOCK_N]` byte tile unpack into two contiguous weight tiles
without a costly register shuffle.

### Kernel

```text
input       [M,K] fp16
packed      [N,K/2] uint8
scales      [N,K/GROUP] fp16
output      [M,N] fp16
grid        (ceil(M/BLOCK_M), ceil(N/BLOCK_N))
```

For each K group:

```python
x_lo = X[:, first half of group]
x_hi = X[:, second half]
packed = load(one byte tile)
low  = (packed & 0xF) - 8
high = (packed >> 4) - 8
scale = scales[output_channel, group]
acc += tl.dot(x_lo, low * scale)
acc += tl.dot(x_hi, high * scale)
```

Accumulation is FP32; output is cast to activation dtype. The model integration replaces
attention, MLP, and output-projection linear modules. A streaming loader constructs modules
on the meta device and loads packed weights without ever assembling the full FP16 model—
necessary for an 8B checkpoint on a ~15 GB card.

### Dense-path slab fallback

Some wider-M uses, such as verification and the large language-model head, may choose a dense
dequantized route. Dequantization is slabbed to bound temporary memory; otherwise unpacking an
8B lm-head can exceed the card even though packed weights fit.

### Tradeoffs

- Roughly 4x lower weight payload before scales; expected to move the bandwidth floor.
- In-kernel nibble unpack/dequantization and two dot products add work.
- Quality and token identity must be judged against the same packed-weight reference, not an
  FP16 model assumed to be numerically identical.
- Current commits include real-checkpoint gates, but final performance conclusions should be
  taken from the newer W4/8B artifacts rather than the older README roadmap text.

---

## 18. W8A16 and W8A8 experiments

### 18.1 W8A16

**File:** `engine/kernels/w8a16_linear.py`  
**Status:** experimental negative result.

Weights use one INT8 value plus one FP16 scale per output channel. The Triton kernel loads
activation and weight tiles, converts weights to FP16, applies the scale, and uses `tl.dot`
with FP32 accumulation.

The idea—halve weight bytes—is sound, but the tested custom path achieved only about
0.12–0.19x the optimized FP16 CUTLASS path on T4. It did not exploit the platform as well
as the mature library GEMM and is not serving-integrated.

### 18.2 W8A8

**File:** `engine/kernels/w8a8_linear.py`  
**Status:** experimental negative result; uses PyTorch CUDA INT8 GEMM, not Triton.

It dynamically quantizes each activation row, calls `torch._int_mm` for INT8×INT8→INT32,
then applies activation and weight scales. Rows below 32 are padded because the CUDA backend
requires a minimum aligned M dimension on tested platforms.

At decode-sized matrices, activation reduction/quantization, temporary INT8 storage, padding,
dequantization, and extra launches outweighed weight savings. The experiment demonstrates why
kernel arithmetic alone is not the cost model.

---

## 19. Backend selection and hardware policy

### Decode backends

| Name | Implementation | Graph-safe? | Key eligibility/status |
|---|---|---:|---|
| `per_head` | K4 Triton | Yes | Default measured path |
| `gqa` | shared two-head Triton | Yes | GQA group exactly 2; negative result |
| `split_k` | partial + merge Triton | Yes | FP16/BF16; useful candidate for narrow grids |
| `flash` | FA2 K/V-cache adapter | No on tested build | sm_80+, page multiple 256, installed wheel |

### Prefill backends

| Name | Implementation | Graph-safe? | Key eligibility/status |
|---|---|---:|---|
| `sdpa` | gather + PyTorch SDPA | Yes | sm_75 measured default; no scaled-pool variant |
| `per_token` | direct paged Triton | Yes | fallback and scaled-pool structure |
| `tiled` | paged FlashAttention-style Triton | Yes | sm_80+; measured sm_89 default |
| `flash` | direct FA2 K/V-cache | No | sm_80+, page multiple 256 |
| `flash_dense` | gather + dense FA2 | No | retained negative-result path |

For INT8/FP8 pools, the engine dispatches the dedicated scaled writer/attention pair rather
than letting an FP16 backend misread quantized storage. Backend availability strings explain
why an option cannot run rather than silently falling back.

Measured policy currently records:

```text
sm_75: decode=per_head, prefill=sdpa, dtype=fp16
sm_89: decode=per_head, prefill=tiled, dtype=fp16
```

An unmeasured GPU uses capability-led priority but is explicitly labeled unmeasured.

---

## 20. Correctness strategy for kernels

The repository uses a ladder of increasingly integrated oracles:

1. **Pure Torch formula/reference** for quantization formats and elementwise operations.
2. **Kernel vs dense mathematical reference** for attention output tensors.
3. **Paged vs contiguous kernel** to isolate page addressing from attention math.
4. **Full engine vs separately loaded stock checkpoint** for greedy token identity.
5. **Pressure, graph, mixed-length, and cancellation tests** for lifecycle interactions.

Important numerical rules:

- Reductions and online-softmax accumulators are FP32.
- Invalid score lanes use `-inf` or a carefully handled finite sentinel.
- Fully masked rows must return zero, not NaN.
- Padding outputs are never sampled or committed.
- A near-tied stock top-2 logit can legitimately flip after numerically equivalent reduction
  orders; the test suite diagnoses margins rather than declaring every late flip a kernel bug.
- A new kernel must pass tensor-level correctness before an end-to-end speed number matters.

Primary tests:

| Topic | Test file |
|---|---|
| RMSNorm | `tests/kernels/test_rmsnorm.py` |
| RoPE/SwiGLU | `tests/kernels/test_fused_ops.py` |
| FP16 KV writers | `tests/kernels/test_kv_write.py` |
| K1 | `tests/kernels/test_triton_attention.py` |
| K2 | `tests/kernels/test_paged_attention_kernel.py` |
| K3 | `tests/kernels/test_paged_kernel_attention.py` |
| K4/GQA/Split-K | `tests/kernels/test_paged_decode_batched.py` |
| Per-token prefill | `tests/kernels/test_paged_prefill.py` |
| SDPA prefill | `tests/kernels/test_sdpa_prefill.py` |
| Tiled prefill | `tests/kernels/test_tiled_paged_prefill.py` |
| INT8 KV | `tests/kernels/test_int8_paged_kv.py` |
| FP8 KV | `tests/kernels/test_fp8_paged_kv.py` |
| W4A16 | `tests/kernels/test_w4a16_linear.py`, `tests/quantization/test_w4a16_cuda.py` |
| W8 paths | `tests/kernels/test_w8a16_linear.py`, `test_w8a8_linear.py` |

---

## 21. Performance-analysis checklist

For every kernel, answer these before claiming it is faster:

1. **What is the launch grid?** Is there enough work to occupy the SMs?
2. **What bytes come from DRAM?** Do source-level duplicate loads actually miss L2?
3. **What lives in registers/shared memory?** Will tile size reduce occupancy or spill?
4. **Does `tl.dot` reach tensor cores?** Inspect PTX; do not infer from source syntax.
5. **What extra work surrounds the kernel?** Gather, masks, workspace, transposes, copies,
   dequantization, or a second merge launch can dominate.
6. **Is it CUDA-graph safe?** Losing graph replay can erase an isolated kernel win.
7. **Does the benchmark treatment actually bind?** A chunk-size experiment is meaningless
   if prompts fit in both settings.
8. **Is the timing harness above its noise/launch floor?** The repository documents that old
   synchronized single-launch microbench cells below roughly 150 µs are not strong evidence.
9. **Does the full engine improve?** The final decision uses repeated, warmed, token-gated
   end-to-end A/Bs, not only an isolated launch.

---

## 22. Interview questions you should be able to answer

1. Why does K4 use one program per `(sequence, query head)` rather than `tl.dot` over a
   large query tile?
2. Derive the page address for logical token 37 with block size 16.
3. Explain `length_offset=1` and why the logical length is committed only after forward.
4. Derive the online-softmax update and the Split-K merge formula.
5. Why did sharing K/V across a GQA pair fail to improve time despite halving explicit loads?
6. Why is gathered SDPA sensible for prefill but not for every decode token?
7. Why did tiled prefill lose on sm_75 and win on sm_89?
8. Why can FlashAttention lose end-to-end even when its isolated attention kernel is faster?
9. Compare internal fragmentation, block-table overhead, and page size.
10. Compare INT8/FP8 KV quantization with W4 weight quantization: which bytes and bottlenecks
    does each target?
11. Why does a fully masked softmax row create NaN, and how do these kernels prevent it?
12. What is graph-safe kernel execution, and why do persistent buffer addresses matter?

---

## 23. Practical study schedule

Use one focused session per row:

| Session | Read | Exercise |
|---:|---|---|
| 1 | Sections 2–3, RMSNorm | Manually calculate strides and one RMSNorm row |
| 2 | RoPE + SwiGLU | Derive paired dimensions and draw both launch grids |
| 3 | KV writers | Trace two tokens crossing a page boundary |
| 4 | K1 + online softmax | Prove tile-update equivalence to ordinary softmax |
| 5 | K2 + K3 | Change only addressing; explain the correctness ladder |
| 6 | K4 | Trace one request/head through two non-contiguous pages |
| 7 | GQA + Split-K | Compare cache reuse, grid width, workspace, merge math |
| 8 | Per-token + SDPA prefill | Calculate repeated-prefix traffic and gather cost |
| 9 | Tiled + Flash prefill | Analyze tiles, masks, tensor-core/PTX and graph safety |
| 10 | INT8 + FP8 KV | Quantize/dequantize one vector and compute storage bytes |
| 11 | W4/W8 linears | Unpack one byte, derive group scaling, compare full cost |
| 12 | Backend policy + engine | Explain exact kernel selection for sm_75 and sm_89 |

For each session, open the implementation and its test side by side. First state the tensor
contract without looking. Then identify the grid, one program’s ownership, every mask, and the
numerical accumulator type. Finally explain why the path is default, optional, or rejected.

