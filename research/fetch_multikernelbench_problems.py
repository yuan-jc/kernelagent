"""Restore pinned MultiKernelBench files from a clean clone; never rewrite manifests.

Two modes:

- default: read the committed configs/multikernelbench manifests and restore
  every listed file into the local (gitignored) research cache, verifying
  sha256 before anything is published.
- --generate-manifest: pin a commit, walk the upstream git tree and download
  the NVIDIA-platform reference problems plus the upstream protocol files,
  emitting a fresh files manifest on stdout. Generation is the only step that
  is allowed to create or replace a manifest, and it never writes problem
  bytes into the repo.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "research/sources/wzzll123__MultiKernelBench"
FILES_MANIFEST = REPO_ROOT / "configs/multikernelbench/snapshot-files.manifest.json"
DEV_MANIFEST = REPO_ROOT / "configs/multikernelbench/dev-manifest.json"

REPOSITORY = "wzzll123/MultiKernelBench"
# NVIDIA-platform problem categories; npukernelbench_* categories target
# Ascend NPUs and are out of scope for this project (recorded, not silently
# mixed into the NVIDIA catalog).
PROBLEM_PREFIX = "reference/"
NPU_CATEGORY = re.compile(r"^reference/npukernelbench")
PROBLEM_SUFFIX = ".py"
# Upstream correctness/timing protocol files pinned beside the problems
# (mirrors configs/kernelbench dev-manifest protocol references).
PROTOCOL_FILES = (
    "config.py",
    "utils/correctness.py",
    "utils/performance.py",
    "utils/utils.py",
    "utils/evaluation_utils.py",
)


def _git_tree_url(commit: str, page: int = 1) -> str:
    return f"https://api.github.com/repos/{REPOSITORY}/git/trees/{commit}?recursive=1&page={page}"


def upstream_problem_paths(commit: str) -> list[str]:
    """List the pinned-commit paths this project snapshots: NVIDIA-platform
    reference problems plus the upstream protocol file closure."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler())
    paths: list[str] = []
    page = 1
    while True:
        with opener.open(_git_tree_url(commit, page), timeout=60) as response:
            tree = json.load(response)
        for item in tree.get("tree", []):
            if item.get("type") != "blob":
                continue
            path = item.get("path", "")
            if path in PROTOCOL_FILES:
                paths.append(path)
            elif path.startswith(PROBLEM_PREFIX) and path.endswith(PROBLEM_SUFFIX):
                if not NPU_CATEGORY.match(path):
                    paths.append(path)
        if tree.get("truncated"):
            raise ValueError("upstream git tree listing was truncated; refusing to snapshot")
        has_next = "link" in response.headers and 'rel="next"' in response.headers["link"]
        if not has_next:
            break
        page += 1
    if not paths:
        raise ValueError("upstream tree produced no snapshot candidates")
    return sorted(set(paths))


def _download(opener, url: str, attempts: int = 4) -> bytes:
    # Upstream TLS endpoints occasionally drop the handshake mid-generation;
    # retry with backoff so one transient error cannot poison a manifest run.
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            with opener.open(url, timeout=60) as response:
                return response.read()
        except (OSError, urllib.error.URLError) as exc:  # includes socket/SSL wrap errors
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise last if last is not None else OSError("download failed")


def generate_manifest(commit: str, workers: int) -> dict:
    """Download every snapshot candidate once and record its content hashes.
    The output schema is identical to configs/kernelbench/snapshot-files.manifest.json."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("commit must be 40-hex")
    paths = upstream_problem_paths(commit)
    base = f"https://raw.githubusercontent.com/{REPOSITORY}/{commit}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler())

    def entry_for(path: str) -> dict:
        data = _download(opener, f"{base}/{path}")
        blob_sha1 = hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()
        return {
            "path": path,
            "git_blob_sha1": blob_sha1,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        files = list(pool.map(entry_for, paths))
    problems = [item for item in files if item["path"].startswith(PROBLEM_PREFIX)]
    return {
        "schema_version": "1.0",
        "repository": REPOSITORY,
        "commit": commit,
        "note": (
            "NVIDIA-platform reference problems only (npukernelbench_* Ascend "
            "categories excluded) plus the upstream protocol file closure."
        ),
        "files": sorted(files, key=lambda item: item["path"]),
        "counts": {"problems": len(problems), "protocol_files": len(files) - len(problems)},
    }


def inventory(files_path: Path, dev_path: Path) -> tuple[str, dict[str, str]]:
    data = files_path.read_bytes()
    files, dev = json.loads(data), json.loads(dev_path.read_bytes())
    upstream = dev["upstream"]
    if files["schema_version"] != "1.0" or dev["schema_version"] != "1.0":
        raise ValueError("unsupported manifest version")
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    if hashlib.sha256(canonical).hexdigest() != upstream["files_manifest_sha256"]:
        raise ValueError("files manifest does not match frozen dev manifest")
    repository, commit = files["repository"], files["commit"]
    if (repository, commit) != (upstream["repository"], upstream["commit"]):
        raise ValueError("manifest repository/commit mismatch")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("invalid upstream repository")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("invalid upstream commit")
    records = list(files["files"])
    if not records:
        raise ValueError("empty problem manifest")
    records += dev["protocol"]["references"]
    entries: dict[str, str] = {}
    for item in records:
        path, digest = item["path"], item["sha256"]
        if (
            not isinstance(path, str)
            or not path
            or "\\" in path
            or ":" in path
            or any(part in ("", ".", "..") for part in path.split("/"))
            or PurePosixPath(path).is_absolute()
        ):
            raise ValueError(f"unsafe manifest path: {path!r}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"invalid sha256: {path}")
        if path in entries and entries[path] != digest:
            raise ValueError(f"conflicting content hashes: {path}")
        entries[path] = digest
    return f"https://raw.githubusercontent.com/{repository}/{commit}", entries


def fetch_one(root: Path, path: str, digest: str, base_url: str, opener) -> None:
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"cache path escapes root: {path}")
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
        return
    with opener.open(f"{base_url}/{path}", timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"download sha256 mismatch: {path}; cache was not changed")
    target.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
            name = Path(tmp.name)
            tmp.write(data)
        os.replace(name, target)
    finally:
        if name is not None:
            name.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--proxy", help="Optional proxy URL; defaults to standard proxy settings")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--generate-manifest",
        action="store_true",
        help="Pin --commit upstream and emit a fresh files manifest on stdout",
    )
    parser.add_argument("--commit", help="40-hex upstream commit for --generate-manifest")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.generate_manifest:
        if not args.commit:
            parser.error("--generate-manifest requires --commit")
        try:
            payload = generate_manifest(args.commit, args.workers)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(json.dumps({"status": "FAIL", "reason": str(exc)}))
            return 1
        print(json.dumps(payload, indent=2))
        return 0
    handler = (
        urllib.request.ProxyHandler({"http": args.proxy, "https": args.proxy})
        if args.proxy
        else urllib.request.ProxyHandler()
    )
    opener = urllib.request.build_opener(handler)
    try:
        base_url, entries = inventory(FILES_MANIFEST, DEV_MANIFEST)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "FAIL", "reason": str(exc)}))
        return 1
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(fetch_one, args.root, path, sha, base_url, opener): path
            for path, sha in entries.items()
        }
        for future in concurrent.futures.as_completed(pending):
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - per-file fetch failures are collected, not swallowed
                failures.append({"path": pending[future], "reason": str(exc)})
    print(
        json.dumps(
            {
                "status": "FAIL" if failures else "PASS",
                "commit": base_url.rsplit("/", 1)[1],
                "files": len(entries),
                "verified": len(entries) - len(failures),
                "root": str(args.root),
                "failures": sorted(failures, key=lambda item: item["path"]),
            },
            indent=2,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
