"""HTTP client for Rastion Hub interactions."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml

from rastion.registry.manager import (
    add_problem,
    hub_config,
    init_registry,
    read_yaml_file,
    set_hub_token,
    set_hub_url,
    solvers_root,
    write_yaml_file,
)

try:  # pragma: no cover - optional dependency in some environments
    import httpx
except ImportError:  # pragma: no cover - handled at runtime in hub commands
    httpx = None  # type: ignore[assignment]

PackageType = Literal["problem", "solver"]
SearchType = Literal["problem", "solver", "both"]


class HubClientError(RuntimeError):
    """Raised when a hub operation fails."""


class HubClient:
    """Client for authentication and package sync with Rastion Hub."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        token: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if httpx is None:
            raise HubClientError("Hub support requires httpx. Install with `pip install httpx`.")

        init_registry(copy_examples=False)
        cfg = hub_config()
        configured_url = str(cfg.get("url") or "http://localhost:8000")

        self.base_url = self._normalize_base_url(base_url or configured_url)
        self.token = token if token is not None else self._normalize_token(cfg.get("token"))
        self.timeout_seconds = timeout_seconds

    def set_base_url(self, url: str, *, persist: bool = True) -> None:
        self.base_url = self._normalize_base_url(url)
        if persist:
            set_hub_url(self.base_url)

    async def login(self, token: str) -> dict[str, Any]:
        candidate = token.strip()
        if not candidate:
            raise HubClientError("GitHub token cannot be empty")

        payload = await self._request_json(
            "POST",
            "/auth/login",
            token_override=candidate,
        )
        access_token = self._extract_access_token(payload)
        user = self._extract_user(payload)
        self.token = access_token
        set_hub_token(access_token)
        return user

    async def logout(self) -> None:
        self.token = None
        set_hub_token(None)

    async def get_user(self) -> dict[str, Any]:
        payload = await self._request_json("GET", "/auth/me", require_auth=True)
        return self._extract_user(payload)

    async def search(self, query: str, type: SearchType = "both") -> dict[str, list[dict[str, Any]]]:
        search_type = self._normalize_search_type(type)
        term = query.strip()

        if search_type == "problem":
            return {"problems": await self._list_items("problems", term), "solvers": []}
        if search_type == "solver":
            return {"problems": [], "solvers": await self._list_items("solvers", term)}

        return {
            "problems": await self._list_items("problems", term),
            "solvers": await self._list_items("solvers", term),
        }

    async def push(self, path: str | Path, type: PackageType | None = None) -> dict[str, Any]:
        source = Path(path).expanduser().resolve()
        if not source.exists() or not source.is_dir():
            raise FileNotFoundError(f"package path does not exist: {source}")

        package_type = self._normalize_package_type(type) if type is not None else self._infer_package_type(source)

        with tempfile.TemporaryDirectory(prefix="rastion_hub_push_") as tmp:
            temp_root = Path(tmp)
            archive_path, manifest = self._build_package_archive(source, package_type, temp_root)

            payload = await self._request_json(
                "POST",
                f"/{package_type}s",
                require_auth=True,
                files={
                    "file": (
                        archive_path.name,
                        archive_path.read_bytes(),
                        "application/zip",
                    )
                },
                data={
                    "name": str(manifest["name"]),
                    "version": str(manifest["version"]),
                    "type": package_type,
                    "manifest": json.dumps(manifest),
                },
            )

        result: dict[str, Any]
        if isinstance(payload, dict):
            result = dict(payload)
        else:
            result = {"response": payload}
        result.setdefault("name", manifest["name"])
        result.setdefault("type", package_type)
        return result

    async def pull(
        self,
        name: str,
        type: SearchType = "both",
        *,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        entry_name = name.strip()
        if not entry_name:
            raise HubClientError("Package name cannot be empty")

        search_type = self._normalize_search_type(type)
        item, package_type = await self._resolve_remote_item(entry_name, search_type)

        download_key = item.get("id") or item.get("name")
        if download_key is None:
            raise HubClientError("Hub response is missing both 'id' and 'name'; cannot download package")

        archive = await self._request_bytes("GET", f"/{package_type}s/{download_key}/download")
        installed_path, installed_name = self._install_archive(
            archive=archive,
            package_type=package_type,
            item=item,
            overwrite=overwrite,
        )

        return {
            "name": installed_name,
            "type": package_type,
            "path": str(installed_path),
            "source": item,
        }

    async def _list_items(self, endpoint: str, query: str) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if query:
            params["q"] = query
        payload = await self._request_json("GET", f"/{endpoint}", params=params or None)
        return self._coerce_items(payload)

    async def _resolve_remote_item(self, name: str, search_type: SearchType) -> tuple[dict[str, Any], PackageType]:
        if search_type == "problem":
            problem = await self._find_best_match("problems", name)
            if problem is None:
                raise HubClientError(f"Problem '{name}' not found on hub")
            return problem, "problem"

        if search_type == "solver":
            solver = await self._find_best_match("solvers", name)
            if solver is None:
                raise HubClientError(f"Solver '{name}' not found on hub")
            return solver, "solver"

        problem = await self._find_best_match("problems", name)
        solver = await self._find_best_match("solvers", name)

        if problem is not None and solver is not None:
            raise HubClientError(
                f"Both problem and solver named '{name}' exist. Retry with an explicit type."
            )
        if problem is not None:
            return problem, "problem"
        if solver is not None:
            return solver, "solver"
        raise HubClientError(f"No problem or solver named '{name}' found on hub")

    async def _find_best_match(self, endpoint: str, name: str) -> dict[str, Any] | None:
        items = await self._list_items(endpoint, name)
        if not items:
            return None

        lowered = name.casefold()
        exact = [item for item in items if str(item.get("name", "")).casefold() == lowered]
        if exact:
            return exact[0]

        startswith = [item for item in items if str(item.get("name", "")).casefold().startswith(lowered)]
        if startswith:
            return startswith[0]

        return items[0]

    def _build_package_archive(
        self,
        source: Path,
        package_type: PackageType,
        temp_root: Path,
    ) -> tuple[Path, dict[str, Any]]:
        if package_type == "problem":
            return self._build_problem_archive(source, temp_root)
        return self._build_solver_archive(source, temp_root)

    def _build_problem_archive(self, source: Path, temp_root: Path) -> tuple[Path, dict[str, Any]]:
        spec_path = source / "spec.json"
        if not spec_path.exists():
            raise FileNotFoundError(f"problem is missing spec.json: {source}")

        metadata = self._load_metadata(source / "metadata.yaml")
        manifest = self._build_manifest(source, metadata, "problem")
        metadata_file = self._build_metadata_file(metadata, manifest)
        readme = self._problem_readme(source, manifest)
        archive_root = self._archive_root_name(manifest["name"], source.name)
        archive_name = f"{archive_root}-{manifest['version']}.zip"
        archive_path = temp_root / archive_name

        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{archive_root}/manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
            zf.write(spec_path, f"{archive_root}/spec.json")

            instances_dir = source / "instances"
            if instances_dir.exists() and instances_dir.is_dir():
                for instance_file in sorted(instances_dir.iterdir()):
                    if not instance_file.is_file() or instance_file.suffix.lower() not in {".json", ".npz"}:
                        continue
                    zf.write(instance_file, f"{archive_root}/instances/{instance_file.name}")
            else:
                legacy_instance = source / "instance.json"
                if legacy_instance.exists():
                    zf.write(legacy_instance, f"{archive_root}/instances/default.json")

            zf.writestr(f"{archive_root}/metadata.yaml", yaml.safe_dump(metadata_file, sort_keys=False))
            zf.writestr(f"{archive_root}/README.md", readme)

        return archive_path, manifest

    def _build_solver_archive(self, source: Path, temp_root: Path) -> tuple[Path, dict[str, Any]]:
        solver_file = self._find_solver_file(source)
        if solver_file is None:
            raise FileNotFoundError(f"solver package must contain solver.py: {source}")

        metadata = self._load_metadata(source / "metadata.yaml")
        manifest = self._build_manifest(source, metadata, "solver")
        metadata_file = self._build_metadata_file(metadata, manifest)
        readme = self._solver_readme(source, manifest)
        archive_root = self._archive_root_name(manifest["name"], source.name)
        archive_name = f"{archive_root}-{manifest['version']}.zip"
        archive_path = temp_root / archive_name

        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{archive_root}/manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
            zf.write(solver_file, f"{archive_root}/solver.py")
            zf.writestr(f"{archive_root}/metadata.yaml", yaml.safe_dump(metadata_file, sort_keys=False))
            zf.writestr(f"{archive_root}/README.md", readme)

        return archive_path, manifest

    def _install_archive(
        self,
        *,
        archive: bytes,
        package_type: PackageType,
        item: dict[str, Any],
        overwrite: bool,
    ) -> tuple[Path, str]:
        with tempfile.TemporaryDirectory(prefix="rastion_hub_pull_") as tmp:
            temp_root = Path(tmp)
            archive_path = temp_root / "package.zip"
            extract_root = temp_root / "extract"
            extract_root.mkdir(parents=True, exist_ok=True)
            archive_path.write_bytes(archive)

            try:
                with zipfile.ZipFile(archive_path) as zf:
                    zf.extractall(extract_root)
            except zipfile.BadZipFile as exc:
                raise HubClientError("Downloaded file is not a valid ZIP archive") from exc

            if package_type == "problem":
                return self._install_problem_from_extract(extract_root, item=item, overwrite=overwrite)
            return self._install_solver_from_extract(extract_root, item=item, overwrite=overwrite)

    def _install_problem_from_extract(
        self,
        extract_root: Path,
        *,
        item: dict[str, Any],
        overwrite: bool,
    ) -> tuple[Path, str]:
        payload_root = self._find_problem_root(extract_root)
        if payload_root is None:
            raise HubClientError("Downloaded archive does not contain a valid problem package")

        manifest = self._load_manifest(payload_root)
        default_name = str(item.get("name") or payload_root.name)
        name = self._safe_name(str(manifest.get("name") or default_name))
        if not name:
            raise HubClientError("Problem package is missing a valid name")

        installed = add_problem(payload_root, name=name, overwrite=overwrite)
        return installed, name

    def _install_solver_from_extract(
        self,
        extract_root: Path,
        *,
        item: dict[str, Any],
        overwrite: bool,
    ) -> tuple[Path, str]:
        payload_root = self._find_solver_root(extract_root)
        if payload_root is None:
            raise HubClientError("Downloaded archive does not contain a valid solver package")

        manifest = self._load_manifest(payload_root)
        default_name = str(item.get("name") or payload_root.name)
        name = self._safe_name(str(manifest.get("name") or default_name))
        if not name:
            raise HubClientError("Solver package is missing a valid name")

        solver_file = self._find_solver_file(payload_root)
        if solver_file is None:
            raise HubClientError("Downloaded solver package does not include solver.py")

        destination = solvers_root() / name
        if destination.exists():
            if not overwrite:
                raise FileExistsError(f"solver '{name}' already exists in registry")
            shutil.rmtree(destination)

        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(solver_file, destination / "solver.py")

        for filename in ("metadata.yaml", "README.md", "manifest.json"):
            source = payload_root / filename
            if source.exists() and source.is_file():
                shutil.copy2(source, destination / filename)

        source_solver_card = payload_root / "solver_card.md"
        if source_solver_card.exists() and source_solver_card.is_file():
            shutil.copy2(source_solver_card, destination / "solver_card.md")

        pulled_at = datetime.now(timezone.utc).isoformat()
        hub_metadata: dict[str, Any] = {
            "name": name,
            "plugin_name": manifest.get("name") if isinstance(manifest.get("name"), str) else name,
            "source": "hub",
            "hub_url": self.base_url,
            "hub_id": item.get("id"),
            "hub_name": item.get("name"),
            "pulled_at": pulled_at,
        }
        write_yaml_file(destination / "hub_source.yaml", hub_metadata)
        return destination, name

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        token_override: str | None = None,
        require_auth: bool = False,
        **kwargs: Any,
    ) -> Any:
        response = await self._request(
            method,
            path,
            token_override=token_override,
            require_auth=require_auth,
            **kwargs,
        )
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise HubClientError(f"Hub response is not valid JSON for {method} {path}") from exc

    async def _request_bytes(
        self,
        method: str,
        path: str,
        *,
        token_override: str | None = None,
        require_auth: bool = False,
        **kwargs: Any,
    ) -> bytes:
        response = await self._request(
            method,
            path,
            token_override=token_override,
            require_auth=require_auth,
            **kwargs,
        )
        return response.content

    async def _request(
        self,
        method: str,
        path: str,
        *,
        token_override: str | None = None,
        require_auth: bool = False,
        **kwargs: Any,
    ) -> httpx.Response:
        headers = dict(kwargs.pop("headers", {}))
        if token_override is not None:
            github_token = token_override.strip()
            if github_token:
                headers.setdefault("Authorization", f"Bearer {github_token}")
                headers.setdefault("X-GitHub-Token", github_token)
        elif self.token:
            headers.setdefault("Authorization", f"Bearer {self.token}")
        elif require_auth:
            raise HubClientError("Not authenticated. Run `rastion login --token <token>` first.")

        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_seconds) as client:
                response = await client.request(method, path, headers=headers, **kwargs)
        except httpx.RequestError as exc:
            raise HubClientError(f"Unable to reach hub at {self.base_url}: {exc}") from exc

        if response.status_code >= 400:
            raise HubClientError(self._format_http_error(response))
        return response

    def _format_http_error(self, response: httpx.Response) -> str:
        detail: str | None = None
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            raw_detail = payload.get("detail")
            if isinstance(raw_detail, str):
                detail = raw_detail
            elif raw_detail is not None:
                detail = json.dumps(raw_detail)
        if not detail:
            detail = response.text.strip() or "unknown error"

        reason = response.reason_phrase or "Error"
        return f"Hub request failed ({response.status_code} {reason}): {detail}"

    def _extract_user(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            user = payload.get("user")
            if isinstance(user, dict):
                return user
            return payload
        raise HubClientError("Hub response did not include user data")

    def _extract_access_token(self, payload: Any) -> str:
        if not isinstance(payload, dict):
            raise HubClientError("Hub response did not include access token")

        access_token = payload.get("access_token")
        if not isinstance(access_token, str) or not access_token.strip():
            raise HubClientError("Hub response did not include access token")
        return access_token.strip()

    def _load_metadata(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        loaded = read_yaml_file(path)
        if isinstance(loaded, dict):
            return dict(loaded)
        return {}

    def _build_metadata_file(self, metadata: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
        result = dict(metadata)
        result.setdefault("name", manifest["name"])
        result.setdefault("version", manifest["version"])
        result.setdefault("author", manifest["author"])
        result.setdefault("description", manifest["description"])
        return result

    def _build_manifest(
        self,
        source: Path,
        metadata: dict[str, Any],
        package_type: PackageType,
    ) -> dict[str, Any]:
        name = self._safe_name(str(metadata.get("name") or source.name))
        if not name:
            raise HubClientError("Package name could not be derived from metadata or folder name")

        version = str(metadata.get("version") or "0.1.0").strip() or "0.1.0"
        description = str(metadata.get("description") or "").strip()
        author = str(metadata.get("author") or "unknown").strip() or "unknown"

        return {
            "name": name,
            "version": version,
            "type": package_type,
            "description": description,
            "author": author,
        }

    def _problem_readme(self, source: Path, manifest: dict[str, Any]) -> str:
        problem_card = source / "problem_card.md"
        if problem_card.exists() and problem_card.is_file():
            return problem_card.read_text(encoding="utf-8")

        readme = source / "README.md"
        if readme.exists() and readme.is_file():
            return readme.read_text(encoding="utf-8")

        return f"# {manifest['name']}\n\nNo problem card provided.\n"

    def _solver_readme(self, source: Path, manifest: dict[str, Any]) -> str:
        solver_card = source / "solver_card.md"
        if solver_card.exists() and solver_card.is_file():
            return solver_card.read_text(encoding="utf-8")

        readme = source / "README.md"
        if readme.exists() and readme.is_file():
            return readme.read_text(encoding="utf-8")

        return f"# {manifest['name']}\n\nNo solver card provided.\n"

    def _infer_package_type(self, source: Path) -> PackageType:
        has_problem = (source / "spec.json").exists()
        has_solver = self._find_solver_file(source) is not None

        if has_problem and has_solver:
            raise HubClientError(
                "Path contains both problem and solver artifacts. Use an explicit type to disambiguate."
            )
        if has_problem:
            return "problem"
        if has_solver:
            return "solver"
        raise HubClientError(
            "Could not infer package type. Expected spec.json (problem) or solver.py (solver)."
        )

    def _find_solver_file(self, source: Path) -> Path | None:
        direct = source / "solver.py"
        if direct.exists() and direct.is_file():
            return direct

        candidates = sorted(
            (path for path in source.rglob("solver.py") if path.is_file()),
            key=lambda path: (len(path.parts), str(path)),
        )
        return candidates[0] if candidates else None

    def _find_problem_root(self, extract_root: Path) -> Path | None:
        candidates = sorted(
            (path.parent for path in extract_root.rglob("spec.json") if path.is_file()),
            key=lambda path: (len(path.parts), str(path)),
        )
        return candidates[0] if candidates else None

    def _find_solver_root(self, extract_root: Path) -> Path | None:
        candidates = sorted(
            (path.parent for path in extract_root.rglob("solver.py") if path.is_file()),
            key=lambda path: (len(path.parts), str(path)),
        )
        return candidates[0] if candidates else None

    def _load_manifest(self, payload_root: Path) -> dict[str, Any]:
        manifest_path = payload_root / "manifest.json"
        if not manifest_path.exists():
            return {}
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HubClientError(f"Invalid manifest.json in pulled package: {exc}") from exc
        if isinstance(loaded, dict):
            return loaded
        return {}

    def _normalize_base_url(self, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise HubClientError("Hub URL cannot be empty")
        return cleaned.rstrip("/")

    def _normalize_search_type(self, value: str) -> SearchType:
        lowered = value.strip().lower()
        if lowered not in {"problem", "solver", "both"}:
            raise HubClientError(f"Unsupported type '{value}'. Expected problem, solver, or both.")
        return lowered  # type: ignore[return-value]

    def _normalize_package_type(self, value: str | None) -> PackageType:
        if value is None:
            raise HubClientError("Package type is required")
        lowered = value.strip().lower()
        if lowered not in {"problem", "solver"}:
            raise HubClientError(f"Unsupported package type '{value}'. Expected problem or solver.")
        return lowered  # type: ignore[return-value]

    def _coerce_items(self, payload: Any) -> list[dict[str, Any]]:
        items: list[Any]
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            if isinstance(payload.get("items"), list):
                items = payload["items"]
            elif isinstance(payload.get("results"), list):
                items = payload["results"]
            elif isinstance(payload.get("problems"), list):
                items = payload["problems"]
            elif isinstance(payload.get("solvers"), list):
                items = payload["solvers"]
            elif isinstance(payload.get("data"), list):
                items = payload["data"]
            else:
                items = []
        else:
            items = []

        return [item for item in items if isinstance(item, dict)]

    def _normalize_token(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _safe_name(self, value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value.strip()).strip("._-")

    def _archive_root_name(self, manifest_name: Any, fallback: str) -> str:
        candidate = self._safe_name(str(manifest_name))
        if candidate:
            return candidate
        fallback_name = self._safe_name(fallback)
        if fallback_name:
            return fallback_name
        return "package"
