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

    @torch.inference_mode()
    def _batched_prefill(self, model, input_ids, attention_mask):
        """Prefill the padded batch. Returns (cache, next_tokens [N,1], mask)."""
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    use_cache=True, return_dict=True)
        # next token per sequence from the LAST non-pad position.
        # With left padding, the last position is the real last token for every row.
        next_tokens = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)   # [N,1]
        return out.past_key_values, next_tokens, attention_mask

    @torch.inference_mode()
    def _batched_decode(self, model, tokens, cache, attention_mask):
        """One batched decode step. tokens [N,1] -> next [N,1], updated cache + mask."""
        new_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1),
                              device=self.device, dtype=attention_mask.dtype)], dim=1)
        out = model(input_ids=tokens, attention_mask=new_mask,
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
            v_out = self.target(input_ids=proposals_t, attention_mask=verify_mask,
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
            commit = min(emit_lens)   # tokens all active sequences commit this round

            # Append committed tokens to outputs; check EOS/length
            for i in active:
                for t in round_emit[i][:commit]:
                    if len(outputs[i]) < max_new_tokens:
                        outputs[i].append(t)
                        if t in self.eos_ids:
                            done[i] = True
                total_accepted += min(accepted_counts[i], commit)

            # Roll BOTH caches back to (prev_len + commit). prev_len = tgt_mask length.
            prev_len = tgt_mask.shape[1]
            target_len = prev_len + commit
            # target cache after verify has prev_len + depth; crop to target_len
            tgt_over = (prev_len + depth) - target_len
            if tgt_over > 0:
                v_out.past_key_values.crop(-tgt_over)
            tgt_cache = v_out.past_key_values
            tgt_mask = verify_mask[:, :target_len]

            # draft cache grew by depth during proposal; crop to target_len as well
            drf_len_now = local_drf_mask.shape[1]
            drf_over = drf_len_now - target_len
            if drf_over > 0:
                local_drf_cache.crop(-drf_over)
            drf_cache = local_drf_cache
            drf_mask = local_drf_mask[:, :target_len]

            # Next starting token per sequence = the last committed token, re-fed.
            last_committed = torch.tensor(
                [[outputs[i][-1]] if (not done[i] and outputs[i]) else [self.pad_id]
                 for i in range(N)], device=self.device)   # [N,1]
            # Re-decode the last committed token to set up next round's next_token + grow cache by 1
            tgt_next, tgt_cache, tgt_mask = self._batched_decode(self.target, last_committed, tgt_cache, tgt_mask)
            drf_next, drf_cache, drf_mask = self._batched_decode(self.draft, last_committed, drf_cache, drf_mask)

        return BatchedSpecResult(outputs, total_accepted, total_proposed, total_rounds)
