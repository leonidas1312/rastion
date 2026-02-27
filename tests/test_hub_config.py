from pathlib import Path

from rastion.registry.manager import (
    config_file,
    init_registry,
    load_config,
    solver_hub_source,
    solvers_root,
    write_yaml_file,
)


def test_load_config_contains_hub_defaults(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "rastion-home"
    monkeypatch.setenv("RASTION_HOME", str(home))

    init_registry(copy_examples=False)
    cfg = load_config()

    assert cfg["hub"]["url"] == "http://localhost:8000"
    assert cfg["hub"]["token"] is None


def test_load_config_backfills_hub_section(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "rastion-home"
    monkeypatch.setenv("RASTION_HOME", str(home))

    init_registry(copy_examples=False)
    write_yaml_file(
        config_file(),
        {
            "schema_version": 1,
            "registry": {
                "provider": "local",
            },
        },
    )

    cfg = load_config()
    assert cfg["hub"]["url"] == "http://localhost:8000"
    assert cfg["hub"]["token"] is None


def test_solver_hub_source_reads_metadata(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "rastion-home"
    monkeypatch.setenv("RASTION_HOME", str(home))

    init_registry(copy_examples=False)
    solver_dir = solvers_root() / "demo_solver"
    solver_dir.mkdir(parents=True, exist_ok=True)
    write_yaml_file(
        solver_dir / "hub_source.yaml",
        {
            "plugin_name": "plugin_alias",
            "hub_url": "http://localhost:8000",
        },
    )

    metadata = solver_hub_source("plugin_alias")
    assert metadata is not None
    assert metadata["folder"] == "demo_solver"
