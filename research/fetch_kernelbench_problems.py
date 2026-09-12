"""Fetch KernelBench problem files at the pinned research commit.

Downloads KernelBench/level{1,2,3,4}/*.py from raw.githubusercontent.com at the
commit recorded in research/SOURCES.md, verifies against git blob SHAs from
tree.json, and writes configs/kernelbench/snapshot-files.manifest.json.

Offline rebuild: `build_source_index.py` recomputes hashes without network.
Re-running this script re-fetches; it must not be presented as a new snapshot
unless the commit changes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = REPO_ROOT / "research" / "sources" / "ScalingIntelligence__KernelBench"
TREE_JSON = SNAPSHOT_DIR / "tree.json"
OUTPUT_MANIFEST = REPO_ROOT / "configs" / "kernelbench" / "snapshot-files.manifest.json"

REPOSITORY = "ScalingIntelligence/KernelBench"
COMMIT = "423217d9fda91e0c2d67e4a43bf62f96f6d104f1"
LEVELS = (1, 2, 3, 4)
RAW_BASE = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}"


def blob_sha1(content: bytes) -> str:
    return hashlib.sha1(b"blob %d\x00" % len(content) + content).hexdigest()


def problem_paths() -> list[tuple[str, str]]:
    tree = json.loads(TREE_JSON.read_text(encoding="utf-8"))
    if tree.get("truncated"):
        raise SystemExit("tree.json is truncated; cannot enumerate all files")
    entries = [
        (entry["path"], entry["sha"])
        for entry in tree["tree"]
        if entry["type"] == "blob"
        and entry["path"].startswith("KernelBench/")
        and any(entry["path"].startswith(f"KernelBench/level{level}/") for level in LEVELS)
        and entry["path"].endswith(".py")
    ]
    return sorted(entries)


def fetch_one(proxy: str | None, path: str, expected_blob: str) -> dict:
    target = SNAPSHOT_DIR / path
    if target.is_file():
        content = target.read_bytes()
        if blob_sha1(content) != expected_blob:
            target.unlink()  # corrupted cache entry; refetch below
    if not target.is_file():
        request = urllib.request.Request(f"{RAW_BASE}/{path}")
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy} if proxy else {})
        )
        with opener.open(request, timeout=60) as response:
            content = response.read()
        actual = blob_sha1(content)
        if actual != expected_blob:
            raise SystemExit(f"blob SHA mismatch for {path}: expected {expected_blob}, got {actual}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    else:
        content = target.read_bytes()
    return {
        "path": path,
        "git_blob_sha1": expected_blob,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", default=None, help="e.g. http://127.0.0.1:7893")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    entries = problem_paths()
    print(f"{len(entries)} problem files at commit {COMMIT[:12]}")
    records: list[dict] = []
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, args.proxy, path, blob): path for path, blob in entries}
        for future in concurrent.futures.as_completed(futures):
            path = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:  # noqa: BLE001 - collect all failures for the summary
                failures.append(f"{path}: {type(exc).__name__}: {exc}")
    if failures:
        print(f"{len(failures)} FAILED:")
        for line in failures:
            print(" ", line)
        return 1
    records.sort(key=lambda item: item["path"])
    manifest = {
        "schema_version": "1.0",
        "repository": REPOSITORY,
        "commit": COMMIT,
        "note": "Problem file inventory at the pinned research commit; files stay in the "
        "local gitignored research/sources cache. The public repo keeps hashes only.",
        "files": records,
    }
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    total = sum(item["size"] for item in records)
    print(f"OK: {len(records)} files verified, {total} bytes; manifest -> {OUTPUT_MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
