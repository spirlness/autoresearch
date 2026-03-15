#!/usr/bin/env python3
from __future__ import annotations

import fnmatch
import subprocess
import sys
from collections import defaultdict

ZERO_OID = "0" * 40
MAX_BLOB_BYTES = 90 * 1024 * 1024
DISALLOWED_PATTERNS = (
    "*.whl",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.safetensors",
    "*.onnx",
)


def run_git(args: list[str], *, stdin: str | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(stderr or f"git {' '.join(args)} failed")
    return result.stdout


def format_bytes(num_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def read_updates() -> list[tuple[str, str, str, str]]:
    updates: list[tuple[str, str, str, str]] = []
    for raw_line in sys.stdin.read().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 4:
            raise RuntimeError(f"unexpected pre-push input line: {raw_line!r}")
        updates.append((parts[0], parts[1], parts[2], parts[3]))
    return updates


def collect_candidate_blobs(remote_name: str, updates: list[tuple[str, str, str, str]]) -> dict[str, set[str]]:
    blob_paths: dict[str, set[str]] = defaultdict(set)
    seen_revisions: set[str] = set()

    for _local_ref, local_oid, _remote_ref, _remote_oid in updates:
        if local_oid == ZERO_OID or local_oid in seen_revisions:
            continue
        seen_revisions.add(local_oid)
        args = ["rev-list", "--objects", local_oid]
        if remote_name:
            args.extend(["--not", f"--remotes={remote_name}"])
        output = run_git(args)
        for line in output.splitlines():
            if " " not in line:
                continue
            obj_id, path = line.split(" ", 1)
            blob_paths[obj_id].add(path)
    return blob_paths


def inspect_blobs(blob_paths: dict[str, set[str]]) -> list[str]:
    if not blob_paths:
        return []

    batch_input = "".join(f"{obj_id}\n" for obj_id in blob_paths)
    output = run_git(
        ["cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
        stdin=batch_input,
    )

    violations: list[str] = []
    for line in output.splitlines():
        obj_id, obj_type, obj_size = line.split()
        if obj_type != "blob":
            continue
        size_bytes = int(obj_size)
        for path in sorted(blob_paths[obj_id]):
            path_lower = path.lower()
            if size_bytes > MAX_BLOB_BYTES:
                violations.append(
                    f"oversized blob: {path} ({format_bytes(size_bytes)} > {format_bytes(MAX_BLOB_BYTES)})"
                )
                continue
            if any(fnmatch.fnmatch(path_lower, pattern) for pattern in DISALLOWED_PATTERNS):
                violations.append(f"tracked artifact type: {path}")
    return violations


def main() -> int:
    remote_name = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        updates = read_updates()
        blob_paths = collect_candidate_blobs(remote_name, updates)
        violations = inspect_blobs(blob_paths)
    except RuntimeError as exc:
        print(f"pre-push check failed: {exc}", file=sys.stderr)
        return 1

    if not violations:
        return 0

    print("Push blocked by repository safety checks.", file=sys.stderr)
    print("", file=sys.stderr)
    for violation in violations:
        print(f"- {violation}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Keep local wheels and model artifacts out of Git history.", file=sys.stderr)
    print("Use vendor/ as an untracked cache, or publish large binaries via releases/object storage.", file=sys.stderr)
    print("If a bad file is already committed, remove it and rewrite history before pushing.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
