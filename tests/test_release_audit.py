from scripts.release_audit import scan_secrets, validate_public_run


def test_secret_scan_ignores_virtualenv(tmp_path):
    venv_file = tmp_path / ".venv" / "lib" / "package.py"
    venv_file.parent.mkdir(parents=True)
    venv_file.write_text("TYPESAFE_API" + "_KEY=not-a-real-secret")

    assert scan_secrets(tmp_path) == []


def test_secret_scan_still_checks_project_files(tmp_path):
    source = tmp_path / "settings.py"
    source.write_text("TYPESAFE_API" + "_KEY=not-a-real-secret")

    assert scan_secrets(tmp_path) == ["settings.py"]


def _valid(name="suite"):
    summary={"model":{"sha256":"m","release_manifest_sha256":"r","run_name":"jev48-1789867911","repo_commit":"c","script_sha256":"s"},"metrics":{"jev48_accuracy":1.0}}
    rows=[{"id":"one","correct":True}]
    return validate_public_run(name,summary,rows,1,lambda row:row["id"],"m","r","c","s")


def test_new_benchmark_validator_accepts_valid_receipt():
    assert _valid() == []


def test_new_benchmark_validator_rejects_metric_tamper():
    summary={"model":{"sha256":"m","release_manifest_sha256":"r","run_name":"jev48-1789867911","repo_commit":"c","script_sha256":"s"},"metrics":{"jev48_accuracy":0.0}}
    errors=validate_public_run("btzsc",summary,[{"id":"one","correct":True}],1,lambda row:row["id"],"m","r","c","s")
    assert "btzsc headline accuracy mismatch" in errors


def test_new_benchmark_validator_rejects_provenance_tamper():
    summary={"model":{"sha256":"wrong","release_manifest_sha256":"r","run_name":"jev48-1789867911","repo_commit":"c","script_sha256":"s"},"metrics":{"jev48_accuracy":1.0}}
    errors=validate_public_run("clash",summary,[{"id":"one","correct":True}],1,lambda row:row["id"],"m","r","c","s")
    assert "clash immutable run provenance mismatch" in errors


def test_new_benchmark_validator_rejects_population_tamper():
    summary={"model":{"sha256":"m","release_manifest_sha256":"r","run_name":"jev48-1789867911","repo_commit":"c","script_sha256":"s"},"metrics":{"jev48_accuracy":1.0}}
    errors=validate_public_run("code-review",summary,[{"id":"same","correct":True},{"id":"same","correct":True}],2,lambda row:row["id"],"m","r","c","s")
    assert "code-review predictions are incomplete or contain duplicate IDs" in errors
