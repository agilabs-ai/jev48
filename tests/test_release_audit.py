from scripts.release_audit import scan_secrets


def test_secret_scan_ignores_virtualenv(tmp_path):
    venv_file = tmp_path / ".venv" / "lib" / "package.py"
    venv_file.parent.mkdir(parents=True)
    venv_file.write_text("TYPESAFE_API" + "_KEY=not-a-real-secret")

    assert scan_secrets(tmp_path) == []


def test_secret_scan_still_checks_project_files(tmp_path):
    source = tmp_path / "settings.py"
    source.write_text("TYPESAFE_API" + "_KEY=not-a-real-secret")

    assert scan_secrets(tmp_path) == ["settings.py"]
