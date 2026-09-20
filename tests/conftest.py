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
    if torch.cuda.is_available():
        import gc

        gc.collect()
        torch.cuda.empty_cache()
