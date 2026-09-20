from pathlib import Path
import tomllib

import jev48


def test_runtime_version_matches_project_metadata():
    metadata = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert jev48.__version__ == metadata["project"]["version"]
