"""Priority 7: per-step fixed overhead at low batch from the engine's instrumentation."""
import json, sys, gc, statistics, time, torch
from engine.model import load_model
from engine.batching.continuous_batching import ContinuousBatchingEngine
from engine.runtime import GenerationRequest
out = {}
for model_name in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"):
    loaded = load_model(model_name, dtype="float16")
    ids = loaded.tokenizer("Paged attention maps logical blocks to physical memory. " * 20, return_tensors="pt").input_ids[0].tolist()[:128]
    for batch in (1, 2, 4):
        eng = ContinuousBatchingEngine(loaded.model, loaded.tokenizer, "cuda", num_blocks=256, max_active=4,
                                       prefix_cache_blocks=0, cuda_graph_batch_sizes=(1, 2, 4))
        eng.warmup(); eng.reset(); eng.eos_ids.clear(); eng.instrument = True
        for i in range(batch):
            eng.submit(GenerationRequest(f"r{i}", len(ids), 160, prompt_token_ids=ids))
        rows = []
        while eng.has_unfinished_requests:
            torch.cuda.synchronize(); t = time.perf_counter(); eng.step(); wall = (time.perf_counter() - t) * 1000
            timing = dict(eng.last_step_timing)
            if eng.last_step_prefill_tokens == 0 and eng.last_step_decode_rows == batch and "decode_gpu_ms" in timing:
                rows.append((wall, timing.get("host_stage_ms", 0), timing.get("decode_gpu_ms", 0), timing.get("sync_ms", 0)))
        rows = rows[10:]
        med = lambda k: statistics.median(r[k] for r in rows)
        # Pure device time of one graph replay, measured back-to-back without host work.
        key = max((k for k in eng._decode_graphs if k[0] == batch), key=lambda k: k[1])
        g = eng._decode_graphs[key]
        torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(50): g.replay()
        e.record(); e.synchronize()
        replay = s.elapsed_time(e) / 50
        row = {"steps": len(rows), "step_wall_ms": med(0), "host_stage_ms": med(1), "decode_event_ms": med(2),
               "sync_ms": med(3), "graph_replay_only_ms": replay, "graph_key": str(key),
               "host_overhead_ms": med(0) - replay, "host_overhead_fraction": (med(0) - replay) / med(0)}
        out[f"{model_name.split('/')[-1]}_b{batch}"] = row
        print(model_name, batch, {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
        del eng; gc.collect(); torch.cuda.empty_cache()
    del loaded; gc.collect(); torch.cuda.empty_cache()
json.dump(out, open(sys.argv[1], "w"), indent=2)
