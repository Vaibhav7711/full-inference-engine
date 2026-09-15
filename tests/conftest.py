import pytest
import torch

from engine.model import LoadedModel, load_model


@pytest.fixture(scope="session")
def loaded() -> LoadedModel:
    """One shared GPU checkpoint for all CUDA integration tests."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    return load_model()
