"""Clean-clone snapshot bootstrap must preserve frozen identities."""

import hashlib
import io
import runpy
from pathlib import Path

import pytest

MODULE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "research/fetch_kernelbench_problems.py")
)


class Download:
    def __init__(self, data):
        self.data = data
        self.calls = 0

    def open(self, url, timeout):
        self.calls += 1
        return io.BytesIO(self.data)


def test_inventory_needs_only_committed_manifests():
    paths = [MODULE["FILES_MANIFEST"], MODULE["DEV_MANIFEST"]]
    before = [path.read_bytes() for path in paths]
    url, entries = MODULE["inventory"](*paths)
    assert url.endswith("423217d9fda91e0c2d67e4a43bf62f96f6d104f1")
    assert "src/kernelbench/dataset.py" in entries
    assert "src/kernelbench/eval.py" in entries
    assert sum(path.startswith("KernelBench/") for path in entries) == 270
    assert [path.read_bytes() for path in paths] == before


def test_download_verifies_bytes_and_reuses_valid_cache(tmp_path):
    data = b"pinned source\n"
    opener = Download(data)
    sha = hashlib.sha256(data).hexdigest()
    fetch = MODULE["fetch_one"]
    fetch(tmp_path, "sub/problem.py", sha, "https://example.invalid/pinned", opener)
    assert (tmp_path / "sub/problem.py").read_bytes() == data
    fetch(tmp_path, "sub/problem.py", sha, "https://example.invalid/pinned", opener)
    assert opener.calls == 1


@pytest.mark.parametrize("existing", [None, b"old corrupt cache"])
def test_bad_download_never_publishes_or_changes_existing_file(tmp_path, existing):
    target = tmp_path / "problem.py"
    if existing is not None:
        target.write_bytes(existing)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        MODULE["fetch_one"](
            tmp_path, "problem.py", "a" * 64, "https://example.invalid", Download(b"wrong")
        )
    assert (target.read_bytes() if target.exists() else None) == existing


def test_fetch_rejects_cache_escape_before_network(tmp_path):
    opener = Download(b"unused")
    with pytest.raises(ValueError, match="escapes root"):
        MODULE["fetch_one"](
            tmp_path / "cache", "../outside.py", "a" * 64, "https://example.invalid", opener
        )
    assert opener.calls == 0
