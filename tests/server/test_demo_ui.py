import asyncio
from pathlib import Path

from fastapi.responses import FileResponse

from engine.server.api import create_app


def test_demo_ui_is_served_at_root() -> None:
    app = create_app(engine_factory=lambda: object())
    route = next(route for route in app.routes if getattr(route, "path", None) == "/")

    response = asyncio.run(route.endpoint())

    assert isinstance(response, FileResponse)
    html = Path(response.path).read_text()
    assert "Inference Engine Lab" in html
    assert "/generate/stream" in html
    assert "/ready" in html
    assert "textContent" in html
