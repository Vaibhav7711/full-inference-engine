import pytest

from engine.model import LoadedModel, load_model


@pytest.fixture(scope="session")
def loaded() -> LoadedModel:
    """One shared GPU checkpoint for all CUDA integration tests."""
    return load_model()
