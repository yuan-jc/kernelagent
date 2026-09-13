"""Benchmark snapshot fetch scripts: manifest inventory and safe restore."""

import hashlib
import io
import runpy
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MKB = runpy.run_path(str(REPO / "research/fetch_multikernelbench_problems.py"))
RK = runpy.run_path(str(REPO / "research/fetch_reference_kernels.py"))


class Download:
    def __init__(self, data):
        self.data = data
        self.calls = 0

    def open(self, url, timeout):
        self.calls += 1
        return io.BytesIO(self.data)


@pytest.mark.parametrize("module", [MKB, RK], ids=["multikernelbench", "reference-kernels"])
def test_inventory_reads_committed_manifests_and_changes_nothing(module):
    paths = [module["FILES_MANIFEST"], module["DEV_MANIFEST"]]
    before = [path.read_bytes() for path in paths]
    url, entries = module["inventory"](*paths)
    assert "raw.githubusercontent.com" in url
    assert len(entries) > 0
    assert all(isinstance(key, str) and isinstance(sha, str) for key, sha in entries.items())
    assert [path.read_bytes() for path in paths] == before


def test_multikernelbench_inventory_covers_nvidia_problems_and_protocol():
    url, entries = MKB["inventory"](MKB["FILES_MANIFEST"], MKB["DEV_MANIFEST"])
    assert url.endswith("460a972c9be7ce18035984321d38b910e27df95f")
    assert sum(path.startswith("reference/") for path in entries) == 300
    for protocol_file in MKB["PROTOCOL_FILES"]:
        assert protocol_file in entries
    assert not any(path.startswith("reference/npukernelbench") for path in entries)


def test_reference_kernels_inventory_is_pmpp_v2_and_linalg_only():
    url, entries = RK["inventory"](RK["FILES_MANIFEST"], RK["DEV_MANIFEST"])
    assert url.endswith("51e22db671d36c1c76091c43c36a44546ba324a1")
    assert len(entries) == 80
    assert "LICENSE" in entries
    assert sum(path.startswith("problems/pmpp_v2/") for path in entries) > 0
    assert sum(path.startswith("problems/linalg/") for path in entries) > 0
    assert not any(path.startswith("problems/helion/") for path in entries)


@pytest.mark.parametrize("module", [MKB, RK], ids=["multikernelbench", "reference-kernels"])
def test_fetch_one_verifies_bytes_and_reuses_valid_cache(module, tmp_path):
    data = b"pinned source\n"
    opener = Download(data)
    sha = hashlib.sha256(data).hexdigest()
    module["fetch_one"](tmp_path, "sub/problem.py", sha, "https://example.invalid/pinned", opener)
    assert (tmp_path / "sub/problem.py").read_bytes() == data
    module["fetch_one"](tmp_path, "sub/problem.py", sha, "https://example.invalid/pinned", opener)
    assert opener.calls == 1


@pytest.mark.parametrize("module", [MKB, RK], ids=["multikernelbench", "reference-kernels"])
@pytest.mark.parametrize("existing", [None, b"old corrupt cache"])
def test_bad_download_never_publishes_or_changes_existing_file(module, tmp_path, existing):
    target = tmp_path / "problem.py"
    if existing is not None:
        target.write_bytes(existing)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        module["fetch_one"](
            tmp_path, "problem.py", "a" * 64, "https://example.invalid", Download(b"wrong")
        )
    assert (target.read_bytes() if target.exists() else None) == existing


@pytest.mark.parametrize("module", [MKB, RK], ids=["multikernelbench", "reference-kernels"])
def test_fetch_rejects_cache_escape_before_network(module, tmp_path):
    opener = Download(b"unused")
    with pytest.raises(ValueError, match="escapes root"):
        module["fetch_one"](
            tmp_path / "cache", "../outside.py", "a" * 64, "https://example.invalid", opener
        )
    assert opener.calls == 0
