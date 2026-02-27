import asyncio
import io
import json
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console

from rastion.hub import client as hub_client_module
from rastion.hub.client import HubClient
from rastion.registry.manager import hub_config, load_config
from rastion.tui import menu

RouteHandler = Callable[[str, dict[str, Any]], "FakeResponse"]


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        json_payload: Any = None,
        content: bytes | None = None,
        reason_phrase: str = "OK",
    ) -> None:
        self.status_code = status_code
        self._json_payload = json_payload
        self.reason_phrase = reason_phrase
        if content is not None:
            self.content = content
        elif json_payload is not None:
            self.content = json.dumps(json_payload).encode("utf-8")
        else:
            self.content = b""

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        if self._json_payload is None:
            raise ValueError("No JSON payload")
        return self._json_payload


def _install_fake_async_client(
    monkeypatch,
    *,
    calls: list[dict[str, Any]],
    routes: dict[tuple[str, str], RouteHandler],
) -> None:
    class FakeAsyncClient:
        def __init__(self, *, base_url: str, timeout: float) -> None:
            self.base_url = str(base_url).rstrip("/")
            self.timeout = timeout

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001, ANN002, ANN003
            return None

        async def request(
            self,
            method: str,
            path: str,
            headers: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> FakeResponse:
            key = (method.upper(), path)
            if key not in routes:
                raise AssertionError(f"Unexpected request: {method} {path}")

            call = {
                "method": method.upper(),
                "path": path,
                "headers": dict(headers or {}),
                "kwargs": kwargs,
            }
            calls.append(call)
            return routes[key](self.base_url, call)

    fake_httpx = type(
        "FakeHttpx",
        (),
        {
            "AsyncClient": FakeAsyncClient,
            "RequestError": RuntimeError,
        },
    )
    monkeypatch.setattr(hub_client_module, "httpx", fake_httpx)


def _json_response(base_url: str, *, method: str, path: str, payload: dict[str, Any]) -> FakeResponse:
    _ = (base_url, method, path)
    return FakeResponse(status_code=200, json_payload=payload)


def _bytes_response(base_url: str, *, method: str, path: str, content: bytes) -> FakeResponse:
    _ = (base_url, method, path)
    return FakeResponse(status_code=200, content=content)


def _solver_zip_bytes(name: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{name}/solver.py", "def solve(*args, **kwargs):\n    return {}\n")
    return buffer.getvalue()


def test_login_persists_jwt_token_from_hub_response(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RASTION_HOME", str(tmp_path / "rastion-home"))

    calls: list[dict[str, Any]] = []
    routes: dict[tuple[str, str], RouteHandler] = {
        ("POST", "/auth/login"): lambda base_url, call: _json_response(
            base_url,
            method=call["method"],
            path=call["path"],
            payload={
                "access_token": "jwt-session-token",
                "user": {"username": "alice"},
            },
        ),
    }
    _install_fake_async_client(monkeypatch, calls=calls, routes=routes)

    client = HubClient(base_url="http://hub.local")
    user = asyncio.run(client.login("gh-token-123"))

    assert user["username"] == "alice"
    assert client.token == "jwt-session-token"
    assert hub_config()["token"] == "jwt-session-token"
    assert calls[0]["headers"]["Authorization"] == "Bearer gh-token-123"
    assert calls[0]["headers"]["X-GitHub-Token"] == "gh-token-123"


def test_push_pull_use_jwt_auth_header_after_login(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RASTION_HOME", str(tmp_path / "rastion-home"))

    calls: list[dict[str, Any]] = []
    routes: dict[tuple[str, str], RouteHandler] = {
        ("POST", "/auth/login"): lambda base_url, call: _json_response(
            base_url,
            method=call["method"],
            path=call["path"],
            payload={
                "access_token": "jwt-after-login",
                "user": {"username": "alice"},
            },
        ),
        ("POST", "/solvers"): lambda base_url, call: _json_response(
            base_url,
            method=call["method"],
            path=call["path"],
            payload={"id": 11, "name": "demo_solver", "type": "solver"},
        ),
        ("GET", "/solvers"): lambda base_url, call: _json_response(
            base_url,
            method=call["method"],
            path=call["path"],
            payload={"items": [{"id": 11, "name": "demo_solver", "version": "0.1.0"}]},
        ),
        ("GET", "/solvers/11/download"): lambda base_url, call: _bytes_response(
            base_url,
            method=call["method"],
            path=call["path"],
            content=_solver_zip_bytes("demo_solver"),
        ),
    }
    _install_fake_async_client(monkeypatch, calls=calls, routes=routes)

    login_client = HubClient(base_url="http://hub.local")
    asyncio.run(login_client.login("github-token"))

    source = tmp_path / "solver_src"
    source.mkdir(parents=True, exist_ok=True)
    (source / "solver.py").write_text("def solve(*args, **kwargs):\n    return {}\n", encoding="utf-8")

    synced_client = HubClient(base_url="http://hub.local")
    asyncio.run(synced_client.push(source, type="solver"))
    pulled = asyncio.run(synced_client.pull("demo_solver", type="solver", overwrite=True))

    assert hub_config()["token"] == "jwt-after-login"
    assert Path(str(pulled["path"])).exists()

    for call in calls[1:]:
        assert call["headers"]["Authorization"] == "Bearer jwt-after-login"
        assert "X-GitHub-Token" not in call["headers"]


def test_tui_login_blank_token_uses_dotenv_fallback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RASTION_HOME", str(tmp_path / "rastion-home"))
    monkeypatch.delenv("RASTION_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("GITHUB_TOKEN=gh-dotenv-token\n", encoding="utf-8")
    monkeypatch.setattr(menu, "_dotenv_candidates", lambda: [dotenv_path])

    used_tokens: list[str] = []

    class FakeHubClient:
        def __init__(self) -> None:
            self.token: str | None = None

        async def login(self, token: str) -> dict[str, Any]:
            used_tokens.append(token)
            self.token = "jwt-from-hub"
            return {"username": "dotenv-user"}

    monkeypatch.setattr(menu, "HubClient", FakeHubClient)
    monkeypatch.setattr(menu, "prompt_input", lambda *args, **kwargs: "")
    monkeypatch.setattr(menu, "wait_for_key", lambda *args, **kwargs: None)
    monkeypatch.setattr(menu.random, "choice", lambda _: "Optimization improves real outcomes.")

    console = Console(record=True, force_terminal=False)
    menu._login_with_github(console)

    cfg = load_config()
    output = console.export_text()
    assert used_tokens == ["gh-dotenv-token"]
    assert cfg["hub"]["token"] == "jwt-from-hub"
    assert cfg["hub"]["username"] == "dotenv-user"
    assert "welcome dotenv-user" in output
    assert "Optimization improves real outcomes." in output
