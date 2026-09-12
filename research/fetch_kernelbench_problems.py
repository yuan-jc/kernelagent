"""Restore pinned KernelBench files from a clean clone; never rewrite manifests."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "research/sources/ScalingIntelligence__KernelBench"
FILES_MANIFEST = REPO_ROOT / "configs/kernelbench/snapshot-files.manifest.json"
DEV_MANIFEST = REPO_ROOT / "configs/kernelbench/dev-manifest.json"


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
    records += dev["protocols"]["upstream_compatible"]["references"]
    records += [{"path": upstream["dataset_module"], "sha256": upstream["dataset_module_sha256"]}]
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
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        base_url, entries = inventory(FILES_MANIFEST, DEV_MANIFEST)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "FAIL", "reason": str(exc)}))
        return 1
    handler = (
        urllib.request.ProxyHandler({"http": args.proxy, "https": args.proxy})
        if args.proxy
        else urllib.request.ProxyHandler()
    )
    opener = urllib.request.build_opener(handler)
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
