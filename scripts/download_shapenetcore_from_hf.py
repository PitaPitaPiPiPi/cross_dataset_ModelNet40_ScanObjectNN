#!/usr/bin/env python3
"""Download and extract ShapeNetCore archives from Hugging Face Datasets."""

import argparse
import fnmatch
import hashlib
import json
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath


SUPPORTED_EXTENSIONS = {".obj", ".off", ".ply", ".npy"}
SYNSET_PATTERN = re.compile(r"^\d{8}$")
MIN_VALIDATION_FILES = 3
VALIDATION_DISPLAY_LIMIT = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download ShapeNetCore zip files from a Hugging Face dataset repo, "
            "extract them, and report the root to pass to scripts.build_shapenet."
        )
    )
    parser.add_argument("--repo_id", default="ShapeNet/ShapeNetCore")
    parser.add_argument("--zip_root", default="raw_datasets/shapenet_zips")
    parser.add_argument("--out_root", default="raw_datasets/shapenet")
    parser.add_argument("--include", default="*.zip")
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--skip_extract", action="store_true")
    parser.add_argument("--remove_zips_after_extract", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def display_path(path):
    text = str(path)
    if any(char.isspace() for char in text):
        return '"' + text.replace('"', '\\"') + '"'
    return text


def print_build_command(shapenet_root):
    root_arg = display_path(shapenet_root)
    print("Next command:")
    print("python -m scripts.build_shapenet \\")
    print(f"  --shapenet_root {root_arg} \\")
    print("  --out_root outputs \\")
    print("  --num_points 1024 \\")
    print("  --workers 4 \\")
    print("  --skip_existing")


def print_plan(args, zip_root, out_root):
    print("Dry run: no directories, downloads, or extracted files will be created.")
    print(f"Repository: {args.repo_id} (dataset)")
    print(f"Include pattern: {args.include}")
    print(f"Zip directory: {zip_root}")
    print(f"Extraction directory: {out_root}")
    print_build_command(out_root)


def download_archives(args, zip_root):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for download. Run: "
            "pip install -U huggingface_hub"
        ) from exc

    print(f"Downloading {args.repo_id} ({args.include}) to {zip_root}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=[args.include],
        local_dir=str(zip_root),
    )


def matches_include(path, zip_root, include_pattern):
    relative = path.relative_to(zip_root).as_posix()
    return fnmatch.fnmatch(relative, include_pattern) or fnmatch.fnmatch(
        path.name, include_pattern
    )


def discover_archives(zip_root, include_pattern):
    if not zip_root.is_dir():
        return []
    return sorted(
        (
            path
            for path in zip_root.rglob("*.zip")
            if path.is_file() and matches_include(path, zip_root, include_pattern)
        ),
        key=lambda path: path.relative_to(zip_root).as_posix().lower(),
    )


def marker_path(archive, zip_root):
    relative = archive.relative_to(zip_root).as_posix()
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()
    return zip_root / ".extract_state" / f"{digest}.json"


def archive_signature(archive):
    file_stat = archive.stat()
    return {"size": file_stat.st_size, "mtime_ns": file_stat.st_mtime_ns}


def marker_matches(marker, archive):
    if not marker.is_file():
        return False
    try:
        with marker.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("archive") == archive_signature(archive)
    except (OSError, ValueError, TypeError):
        return False


def write_marker(marker, archive):
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open("w", encoding="utf-8") as f:
        json.dump({"archive": archive_signature(archive)}, f, indent=2)


def archive_synset_ids(infos):
    synsets = set()
    for info in infos:
        path = PurePosixPath(info.filename.replace("\\", "/"))
        synsets.update(part for part in path.parts if SYNSET_PATTERN.fullmatch(part))
    return synsets


def archive_name_synset_id(archive):
    match = re.search(r"(?<!\d)(\d{8})(?!\d)", archive.stem)
    return match.group(1) if match else None


def contains_supported_file(root):
    if not root.is_dir():
        return False
    return any(
        path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        for path in root.rglob("*")
    )


def find_existing_synset(out_root, synset_id):
    direct = out_root / synset_id
    if contains_supported_file(direct):
        return direct
    if out_root.is_dir():
        for child in out_root.iterdir():
            nested = child / synset_id
            if child.is_dir() and contains_supported_file(nested):
                return nested
    return None


def archive_already_extracted(out_root, infos, archive):
    synsets = archive_synset_ids(infos)
    name_synset = archive_name_synset_id(archive)
    if name_synset:
        synsets.add(name_synset)
    return bool(synsets) and all(
        find_existing_synset(out_root, synset_id) is not None
        for synset_id in synsets
    )


def validate_zip_member(info, out_root):
    member_path = PurePosixPath(info.filename.replace("\\", "/"))
    if (
        member_path.is_absolute()
        or ".." in member_path.parts
        or (member_path.parts and member_path.parts[0].endswith(":"))
    ):
        raise ValueError(f"Unsafe zip member path: {info.filename}")

    file_type = (info.external_attr >> 16) & 0o170000
    if file_type == stat.S_IFLNK:
        raise ValueError(f"Symbolic link is not allowed in zip: {info.filename}")

    target = out_root.joinpath(*member_path.parts).resolve()
    root = out_root.resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Zip member escapes extraction root: {info.filename}")


def extract_archive(archive, zip_root, out_root, overwrite):
    marker = marker_path(archive, zip_root)
    with zipfile.ZipFile(archive, "r") as zip_file:
        infos = zip_file.infolist()
        member_synsets = archive_synset_ids(infos)
        name_synset = archive_name_synset_id(archive)
        extraction_root = (
            out_root / name_synset
            if not member_synsets and name_synset
            else out_root
        )
        if not overwrite and (
            marker_matches(marker, archive)
            or archive_already_extracted(out_root, infos, archive)
        ):
            if not marker_matches(marker, archive):
                write_marker(marker, archive)
            return "skipped"

        for info in infos:
            validate_zip_member(info, extraction_root)
        extraction_root.mkdir(parents=True, exist_ok=True)
        for info in infos:
            zip_file.extract(info, path=extraction_root)

    write_marker(marker, archive)
    return "extracted"


def direct_synset_dirs(root):
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and SYNSET_PATTERN.fullmatch(path.name)
    )


def infer_shapenet_root(out_root):
    direct = direct_synset_dirs(out_root)
    if len(direct) >= 2:
        return out_root, f"found {len(direct)} synset directories directly below it"

    nested_candidates = []
    if out_root.is_dir():
        for child in out_root.iterdir():
            if child.is_dir():
                count = len(direct_synset_dirs(child))
                if count >= 2:
                    nested_candidates.append((count, child))

    if nested_candidates:
        nested_candidates.sort(key=lambda item: (-item[0], str(item[1]).lower()))
        count, candidate = nested_candidates[0]
        return candidate, f"found {count} synset directories one level below out_root"

    return out_root, "no directory containing multiple synset IDs was detected"


def find_validation_samples(out_root, limit=VALIDATION_DISPLAY_LIMIT):
    samples = []
    if not out_root.is_dir():
        return samples
    for path in out_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            samples.append(path)
            if len(samples) >= limit:
                break
    return samples


def print_validation(out_root):
    samples = find_validation_samples(out_root)
    if len(samples) < MIN_VALIDATION_FILES:
        print(
            f"Validation failed: found only {len(samples)} supported files under "
            f"{out_root}; expected at least {MIN_VALIDATION_FILES}."
        )
        return False

    suffixes = sorted({path.suffix.lower() for path in samples})
    qualifier = ">=" if len(samples) == VALIDATION_DISPLAY_LIMIT else "="
    print(
        f"Validation passed: supported files {qualifier}{len(samples)} "
        f"(sampled extensions: {', '.join(suffixes)})"
    )
    return True


def print_root_recommendation(out_root):
    recommended_root, reason = infer_shapenet_root(out_root)
    print(f"Recommended --shapenet_root: {recommended_root}")
    print(f"Reason: {reason}.")
    print_build_command(recommended_root)
    return recommended_root


def run(args):
    zip_root = Path(args.zip_root).expanduser()
    out_root = Path(args.out_root).expanduser()

    if args.dry_run:
        print_plan(args, zip_root, out_root)
        return 0

    if not args.skip_download:
        zip_root.mkdir(parents=True, exist_ok=True)
        try:
            download_archives(args, zip_root)
        except Exception as exc:
            print(f"Download failed: {type(exc).__name__}: {exc}")
            print("Confirm dataset access approval and run `hf auth login`.")
            return 1
    else:
        print(f"Skip download; use existing archives in {zip_root}")

    archives = discover_archives(zip_root, args.include)
    print(f"Zip archives found: {len(archives)}")
    if not archives:
        print(f"No zip files matching {args.include!r} found under {zip_root}")
        print_root_recommendation(out_root)
        return 1

    if args.skip_extract:
        print("Skip extraction; downloaded zip files were kept.")
        print_root_recommendation(out_root)
        return 0

    extracted = []
    skipped = []
    removed = []
    failures = []
    for index, archive in enumerate(archives, start=1):
        relative = archive.relative_to(zip_root)
        print(f"[{index}/{len(archives)}] Extract {relative}")
        try:
            result = extract_archive(
                archive,
                zip_root,
                out_root,
                overwrite=args.overwrite,
            )
            if result == "skipped":
                skipped.append(relative)
                print(f"  Skip existing extraction: {relative}")
            else:
                extracted.append(relative)
                print(f"  Extracted: {relative}")
                if args.remove_zips_after_extract:
                    archive.unlink()
                    removed.append(relative)
                    print(f"  Removed zip: {relative}")
        except Exception as exc:
            failures.append((relative, f"{type(exc).__name__}: {exc}"))
            print(f"  Failed: {relative}: {type(exc).__name__}: {exc}")

    print("Summary:")
    print(f"  extracted: {len(extracted)}")
    print(f"  skipped existing: {len(skipped)}")
    print(f"  removed zips: {len(removed)}")
    print(f"  failed: {len(failures)}")
    if failures:
        print("Failed zip files:")
        for relative, error in failures:
            print(f"  - {relative}: {error}")

    validation_ok = print_validation(out_root)
    print_root_recommendation(out_root)
    return 1 if failures or not validation_ok else 0


def main():
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
