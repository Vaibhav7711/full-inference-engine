__all__ = ["create_app"]


def create_app(*args, **kwargs):
    """Lazily import the GPU-backed FastAPI surface."""
    from .api import create_app as _create_app
    return _create_app(*args, **kwargs)
