"""Batched speculative decoding — N sequences speculate together in one padded batch.

The capstone: combine speculative decoding with batching. N sequences each run greedy
speculation, but the draft and target forward passes are BATCHED across all N sequences,
so the model weights are read once per step and amortized — the batching win — while
speculation attempts to produce multiple tokens per step.

The research question this answers: does speculation still help as batch size grows?
Batching fills the GPU's idle compute (that's how it gets throughput); speculation also
needs idle compute (to verify cheaply). They compete. This engine lets you MEASURE the
crossover where speculation stops helping.

Design — padded batch (buildable, correct, no custom kernel):
    - All N active sequences in ONE left-padded batch with an attention mask.
    - HF's native batched forward handles [N, ...] tensors with the mask.
    - Per round:
        1. DRAFT: draft model generates K tokens for all N sequences (K batched forwards).
        2. VERIFY: target model checks the K proposals for all N (batched).
        3. ACCEPT (ragged): each sequence accepts a DIFFERENT prefix length.
        4. ROLLBACK (ragged): each sequence's cache is cropped to its accepted length;
           since sequences accept different amounts, we crop the batch to the MINIMUM
           accepted and re-queue the rest (simple, correct), OR track per-sequence lengths.
    - This reuses HF's proven padded-batch + crop, which is how assisted generation batches.

Correctness gate: batched speculative output == per-sequence vanilla speculative output.

Honest note on the experiment: this measures batched-forward speculation. The expected
finding (from the literature) is that speculation's benefit shrinks as batch grows,
because batching consumes the idle compute speculation relies on. Whichever way it lands,
it's a real measured result on your own engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import DynamicCache


@dataclass
class BatchedSpecResult:
    outputs: list[list[int]]         # output token ids per sequence
    total_accepted: int
    total_proposed: int
    total_rounds: int

    @property
    def acceptance_rate(self) -> float:
        return self.total_accepted / self.total_proposed if self.total_proposed else 0.0


class BatchedSpeculativeEngine:
    """Greedy speculative decoding over a padded batch of N sequences."""

    def __init__(self, target, draft, tokenizer, device):
        self.target = target.eval()
        self.draft = draft.eval()
        self.tok = tokenizer
        self.device = device
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        eos = target.generation_config.eos_token_id or tokenizer.eos_token_id
        self.eos_ids = {eos} if isinstance(eos, int) else set(eos)

    @staticmethod
    def _position_ids_from_mask(attention_mask):
        """Correct position_ids for a (left-padded) batch: cumsum of the mask - 1.

        Left-padded rows have pads on the left; their real tokens must be positioned 0,1,2..
        regardless of pad count. cumsum(mask)-1 gives exactly that; pads get a dummy pos.
        """
        pos = attention_mask.long().cumsum(-1) - 1
        pos.masked_fill_(attention_mask == 0, 1)
        return pos

    @torch.inference_mode()
    def _batched_prefill(self, model, input_ids, attention_mask):
        """Prefill the padded batch. Returns (cache, next_tokens [N,1], mask)."""
        position_ids = self._position_ids_from_mask(attention_mask)
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    position_ids=position_ids, use_cache=True, return_dict=True)
        # next token per sequence from the LAST position (right-most = real last token,
        # since left padding puts reals on the right).
        next_tokens = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)   # [N,1]
        return out.past_key_values, next_tokens, attention_mask

    @torch.inference_mode()
    def _batched_decode(self, model, tokens, cache, attention_mask):
        """One batched decode step. tokens [N,1] -> next [N,1], updated cache + mask.

        Passes per-row position_ids so left-padded sequences keep correct RoPE positions.
        The new token's position for each row = (real tokens so far) = cumsum(new_mask)-1
        at the last column.
        """
        new_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1),
                              device=self.device, dtype=attention_mask.dtype)], dim=1)
        # position of the new token per row = number of real tokens so far - 1... actually
        # the NEW token's position = (count of real tokens INCLUDING it) - 1 = cumsum-1 at last col.
        pos_full = new_mask.long().cumsum(-1) - 1            # [N, L]
        position_ids = pos_full[:, -1:].clamp(min=0)         # [N,1] position of the new token
        out = model(input_ids=tokens, attention_mask=new_mask, position_ids=position_ids,
                    past_key_values=cache, use_cache=True, return_dict=True)
        next_tokens = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        return next_tokens, out.past_key_values, new_mask

    @torch.inference_mode()
    def generate(self, prompts: list[str], *, max_new_tokens: int, speculation_depth: int = 4) -> BatchedSpecResult:
        N = len(prompts)
        # Left-pad the prompts into one batch
        self.tok.padding_side = "left"
        enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
        input_ids, attn = enc.input_ids, enc.attention_mask

        # Prefill both models on the padded batch
        tgt_cache, tgt_next, tgt_mask = self._batched_prefill(self.target, input_ids, attn)
        drf_cache, drf_next, drf_mask = self._batched_prefill(self.draft, input_ids, attn)

        outputs: list[list[int]] = [[] for _ in range(N)]
        done = [False] * N
        total_accepted = total_proposed = total_rounds = 0

        # We advance all sequences in lockstep. A sequence that's "done" still rides along
        # in the batch (its tokens are ignored) until all finish — simplest correct batching.
        while not all(done) and max(len(o) for o in outputs) < max_new_tokens:
            depth = speculation_depth
            total_rounds += 1

            # --- 1. DRAFT: K batched decode steps on the draft ---
            # proposals[k] is [N,1] — the k-th proposed token for each sequence.
            proposals = []
            cur = drf_next
            local_drf_cache, local_drf_mask = drf_cache, drf_mask
            for _ in range(depth):
                proposals.append(cur)                                    # [N,1]
                cur, local_drf_cache, local_drf_mask = self._batched_decode(
                    self.draft, cur, local_drf_cache, local_drf_mask)
            proposals_t = torch.cat(proposals, dim=1)                    # [N, depth]
            total_proposed += depth * N

            # --- 2. VERIFY: feed the target the depth proposals in ONE batched forward ---
            # Target sees [N, depth] and predicts the token AFTER each position.
            verify_mask = torch.cat([tgt_mask, torch.ones((N, depth), device=self.device,
                                     dtype=tgt_mask.dtype)], dim=1)
            # position_ids for the `depth` verify tokens per row (left-padding aware).
            v_pos_full = verify_mask.long().cumsum(-1) - 1        # [N, L]
            verify_pos = v_pos_full[:, -depth:].clamp(min=0)      # [N, depth]
            v_out = self.target(input_ids=proposals_t, attention_mask=verify_mask,
                                position_ids=verify_pos,
                                past_key_values=tgt_cache, use_cache=True, return_dict=True)
            # target prediction at position j is argmax of logits[:, j, :]
            # prediction[0] should match proposals[0] if draft agreed; etc.
            # The target's "correct" token sequence: [tgt_next, then argmax of each pos]
            v_logits = v_out.logits                                       # [N, depth, vocab]
            tgt_preds = v_logits.argmax(dim=-1)                          # [N, depth]
            # The token the target wants at position 0 is tgt_next (the prev step's pick);
            # positions 1..depth-1 come from v_logits[:, 0..depth-2]. Bonus = v_logits[:,-1].
            # Build target's greedy chain per sequence:
            #   want[0] = tgt_next[i], want[j] = argmax(v_logits[i, j-1]) for j>=1
            want = torch.cat([tgt_next, tgt_preds[:, :-1]], dim=1)        # [N, depth]
            bonus = tgt_preds[:, -1]                                      # [N]

            # --- ONE sync: pull proposals, want, bonus for per-sequence Python accept ---
            proposals_list = proposals_t.tolist()   # [N][depth]
            want_list = want.tolist()               # [N][depth]
            bonus_list = bonus.tolist()             # [N]

            # --- 3. ACCEPT per sequence (ragged) ---
            accepted_counts = []
            round_emit = []   # tokens emitted this round per sequence
            for i in range(N):
                if done[i]:
                    accepted_counts.append(0)
                    round_emit.append([])
                    continue
                acc = 0
                for j in range(depth):
                    if proposals_list[i][j] != want_list[i][j]:
                        break
                    acc += 1
                if acc == depth:
                    emit = proposals_list[i] + [bonus_list[i]]
                else:
                    emit = proposals_list[i][:acc] + [want_list[i][acc]]
                accepted_counts.append(acc)
                round_emit.append(emit)

            # --- 4. COMMIT + ROLLBACK (ragged via crop to committed length) ---
            # Each sequence commits len(emit) tokens. But the batch's caches grew by `depth`
            # (target) and `depth` (draft). We must roll back to a CONSISTENT batch length.
            # Simplest correct approach: commit the MINIMUM emit length across active seqs,
            # crop both caches to that, and carry leftover emitted tokens forward as the
            # next round's starting tokens. This keeps the batch rectangular (one length).
            #
            # committed_len[i] = number of tokens sequence i actually keeps this round.
            active = [i for i in range(N) if not done[i]]
            if not active:
                break
            emit_lens = [len(round_emit[i]) for i in active]
            commit = min(emit_lens)   # uniform rectangular commit length this round
            # commit >= 1 always (every sequence emits at least the correction/bonus token).

            # Append the committed (min) tokens to each active sequence's output.
            # Sequences that accepted MORE re-propose the rest next round (correct, just
            # more rounds — no tokens lost, still greedy-equivalent).
            for i in active:
                for t in round_emit[i][:commit]:
                    if len(outputs[i]) < max_new_tokens and not done[i]:
                        outputs[i].append(t)
                        if t in self.eos_ids:
                            done[i] = True
                total_accepted += min(accepted_counts[i], commit)

            prev_len = tgt_mask.shape[1]

            # THE FIX: crop BOTH caches to prev_len + commit - 1 (one short), then decode the
            # last committed token ONCE. This adds the last committed token exactly once
            # (avoiding the double-count that corrupted the cache before). The decode's
            # output gives next_token for the next round, and the cache ends at prev_len+commit.
            crop_len = prev_len + commit - 1

            tgt_over = (prev_len + depth) - crop_len
            if tgt_over > 0:
                v_out.past_key_values.crop(-tgt_over)
            tgt_cache = v_out.past_key_values
            tgt_mask = verify_mask[:, :crop_len]

            drf_len_now = local_drf_mask.shape[1]
            drf_over = drf_len_now - crop_len
            if drf_over > 0:
                local_drf_cache.crop(-drf_over)
            drf_cache = local_drf_cache
            drf_mask = local_drf_mask[:, :crop_len]

            # The last committed token per sequence = round_emit[i][commit-1].
            last_committed = torch.tensor(
                [[round_emit[i][commit - 1]] if i in active else [self.pad_id]
                 for i in range(N)], device=self.device)   # [N,1]
            # Decode it ONCE: adds it to cache (now at prev_len+commit) and yields next_token.
            tgt_next, tgt_cache, tgt_mask = self._batched_decode(self.target, last_committed, tgt_cache, tgt_mask)
            drf_next, drf_cache, drf_mask = self._batched_decode(self.draft, last_committed, drf_cache, drf_mask)

        return BatchedSpecResult(outputs, total_accepted, total_proposed, total_rounds)
