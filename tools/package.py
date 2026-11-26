#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Create a deterministic source archive with an internal SHA-256 manifest."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = {"README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "CMakeLists.txt",
             "Dockerfile", "pyproject.toml", "bench", ".gitignore", ".dockerignore"}
DIRECTORIES = {"src", "include", "python", "tests", "tools", "configs", "cmake", "examples", "docs", "validation", "reports", ".github"}
SUFFIXES = {".cu", ".cuh", ".cpp", ".hpp", ".h", ".inc", ".cmake", ".py", ".json", ".jsonl",
            ".md", ".txt", ".log", ".yml", ".yaml", ".html", ".csv", ".sass", ".gz", ".diff", ".patch"}


def source_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or any(part.endswith(".egg-info") for part in relative.parts):
            continue
        if (len(relative.parts) == 1 and relative.name in TOP_LEVEL) or (
            len(relative.parts) > 1 and relative.parts[0] in DIRECTORIES and path.suffix in SUFFIXES
        ):
            yield path, relative


def build_archive(output: Path):
    output = output.resolve()
    if output.exists():
        raise ValueError("archive already exists; choose a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    manifest = {"schema_version": 1, "project": "ArchBench", "version": "0.2.0",
                "meaning": "Source package integrity; not a GPU validation attestation.", "files": []}
    for path, relative in source_files(ROOT):
        contents = path.read_bytes()
        name = "archbench/" + relative.as_posix()
        executable = bool(path.stat().st_mode & stat.S_IXUSR)
        mode = 0o100755 if executable else 0o100644
        manifest["files"].append({"path": name, "bytes": len(contents),
                                  "sha256": hashlib.sha256(contents).hexdigest()})
        entries.append((name, contents, mode))
    entries.append(("archbench/SOURCE_MANIFEST.json", (json.dumps(manifest, indent=2) + "\n").encode(), 0o100644))
    # Stable metadata makes the same input bytes produce the same source ZIP.
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, contents, mode in sorted(entries):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 25, 0, 0, 0))
            info.create_system = 3
            info.external_attr = mode << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, contents, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("archive CRC validation failed")
        for record in manifest["files"]:
            contents = archive.read(record["path"])
            if len(contents) != record["bytes"] or hashlib.sha256(contents).hexdigest() != record["sha256"]:
                raise ValueError("archive manifest validation failed")
    return {"archive": str(output), "source_files": len(manifest["files"]),
            "uncompressed_bytes": sum(record["bytes"] for record in manifest["files"]),
            "archive_bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "crc_and_manifest_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(build_archive(args.output), indent=2))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
