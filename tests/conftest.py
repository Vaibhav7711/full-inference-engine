import pytest
import torch

from engine.model import LoadedModel, load_model


@pytest.fixture(scope="session")
def loaded() -> LoadedModel:
    """One shared GPU checkpoint for all CUDA integration tests."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    return load_model()


@pytest.fixture(autouse=True)
def _release_cuda_between_tests():
    """Return every test's GPU allocations to the driver before the next test loads.

    A test's locals die when it returns, but the caching allocator keeps the blocks and
    the next model load asks for fresh ones. Over a hundred CUDA tests that compounds
    into an OOM in whichever test happens to load last.
    """
    yield
    # The engine publishes per-step attention metadata through module globals. A test
    # that fails inside a step leaves them set, and they reference the KV pools - so
    # clear them before collecting, or the pools outlive the test that made them.
    try:
        from engine.batching import continuous_batching as cb

        cb._clear_batch_ctx()
        cb._clear_prefill_ctx()
        cb._clear_fused_ctx()
    except Exception:  # pragma: no cover - the module needs Triton to import
        pass
    if torch.cuda.is_available():
        import gc

        gc.collect()
        torch.cuda.empty_cache()
