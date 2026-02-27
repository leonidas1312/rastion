from rastion.cli.main import main


def test_info_verbose_reports_missing_capabilities(capsys) -> None:
    exit_code = main(
        [
            "info",
            "--verbose",
            "examples/maxcut/spec.json",
            "examples/maxcut/instance.json",
        ]
    )
    assert exit_code == 0

    output = capsys.readouterr().out
    assert "- baseline: NO MATCH" in output
    assert "missing: objective_type: requires qubo" in output
