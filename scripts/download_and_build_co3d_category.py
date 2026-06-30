#!/usr/bin/env python3
# python -m scripts.download_and_build_co3d_category --co3d_repo external/co3d --download_folder raw_datasets/co3d --out_root outputs --category apple --num_points 1024 --workers 4 --cleanup_raw
import argparse
import os
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

from scripts.build_co3d import (
    build_category,
    parse_categories,
    parse_cross_categories,
    rebuild_split_manifest_from_outputs,
)
from scripts.utils.logger import get_logger


SAFE_DOWNLOADER_ERROR = (
    "現在の CO3D downloader ではカテゴリ指定 download が確認できないため、"
    "安全のため一括 download は実行しない"
)
CATEGORY_OPTION_CANDIDATES = (
    "--download_categories",
    "--download_category",
    "--categories",
    "--category",
)
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz")


def resolve_category_targets(args):
    target_sources = [
        bool(args.category),
        bool(args.categories),
        bool(args.all_categories),
        bool(args.cross_classes_only),
    ]
    if sum(target_sources) != 1:
        raise ValueError(
            "Specify exactly one of --category, --categories, --all_categories, "
            "or --cross_classes_only"
        )
    if args.all_categories:
        return parse_categories(None)
    if args.cross_classes_only:
        return parse_cross_categories()
    return parse_categories(args.category or args.categories)


def category_output_exists(out_root, category):
    category_dir = Path(out_root) / "CO3D" / category
    return any((category_dir / "train").glob("*.npy")) or any(
        (category_dir / "test").glob("*.npy")
    )


def category_downloaded(download_folder, category):
    category_dir = Path(download_folder) / category
    if not category_dir.exists():
        return False
    return any(category_dir.glob("*/pointcloud.ply"))


def ensure_under_root(path, root):
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        common = os.path.commonpath([str(path), str(root)])
    except ValueError:
        return False
    return common == str(root) and path != root


def remove_category_output(out_root, category):
    logger = get_logger("download_and_build_co3d_category")
    category_dir = Path(out_root) / "CO3D" / category
    co3d_out_root = Path(out_root) / "CO3D"
    if not category_dir.exists():
        return
    if not ensure_under_root(category_dir, co3d_out_root):
        raise RuntimeError(f"Refusing to remove unsafe output path: {category_dir}")
    logger.warning(f"Overwrite: remove existing output {category_dir}")
    shutil.rmtree(category_dir)


def downloader_path(co3d_repo):
    co3d_repo = Path(co3d_repo)
    candidates = [
        co3d_repo / "co3d" / "download_dataset.py",  # CO3Dv2 branch layout
        co3d_repo / "download_dataset.py",  # CO3Dv1 branch layout
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "CO3D downloader not found. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def downloader_help(co3d_repo):
    downloader = downloader_path(co3d_repo)
    cmd = [sys.executable, str(downloader), "--help"]
    completed = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0 and not completed.stdout:
        raise RuntimeError(f"Failed to inspect CO3D downloader help: {downloader}")
    return completed.stdout or ""


def detect_downloader_options(co3d_repo):
    help_text = downloader_help(co3d_repo)
    category_option = None
    for option in CATEGORY_OPTION_CANDIDATES:
        if option in help_text:
            category_option = option
            break
    if category_option is None:
        raise RuntimeError(SAFE_DOWNLOADER_ERROR)
    return {
        "category_option": category_option,
        "has_single_sequence_subset": "--single_sequence_subset" in help_text,
        "has_clear_archives_after_unpacking": "--clear_archives_after_unpacking" in help_text,
        "has_redownload_existing_archives": "--redownload_existing_archives" in help_text,
        "has_checksum_check": "--checksum_check" in help_text,
    }


def run_downloader(args, category, downloader_options):
    logger = get_logger("download_and_build_co3d_category")
    downloader = downloader_path(args.co3d_repo)
    cmd = [
        sys.executable,
        str(downloader),
        "--download_folder",
        str(args.download_folder),
        downloader_options["category_option"],
        category,
    ]
    if args.single_sequence_subset:
        if downloader_options["has_single_sequence_subset"]:
            cmd.append("--single_sequence_subset")
        else:
            logger.warning("Downloader help does not expose --single_sequence_subset.")
    if args.clear_archives_after_unpacking:
        if downloader_options["has_clear_archives_after_unpacking"]:
            cmd.append("--clear_archives_after_unpacking")
        else:
            logger.warning("Downloader help does not expose --clear_archives_after_unpacking.")
    if downloader_options["has_redownload_existing_archives"]:
        cmd.append("--redownload_existing_archives")
    if args.checksum_check:
        if downloader_options["has_checksum_check"]:
            cmd.append("--checksum_check")
        else:
            logger.warning("Downloader help does not expose --checksum_check.")

    logger.info(f"Download CO3D: category={category}")
    subprocess.run(cmd, check=True)


def get_downloader_options(args, cache):
    if cache.get("value") is None:
        options = detect_downloader_options(args.co3d_repo)
        cache["value"] = options
        get_logger("download_and_build_co3d_category").info(
            f"Downloader category option: {options['category_option']}"
        )
    return cache["value"]


def process_category(args, category):
    process_args = Namespace(
        co3d_root=args.download_folder,
        out_root=args.out_root,
        categories=category,
        num_points=args.num_points,
        train_ratio=0.8,
        workers=args.workers,
        seed=args.seed,
        use_set_lists=False,
        skip_existing=args.skip_existing and not args.overwrite,
        overwrite=args.overwrite,
        strict=True,
    )
    build_category(category, process_args)


def validate_category_output(out_root, category, num_points):
    category_dir = Path(out_root) / "CO3D" / category
    files = sorted((category_dir / "train").glob("*.npy")) + sorted(
        (category_dir / "test").glob("*.npy")
    )
    if not files:
        raise RuntimeError(f"No processed .npy files found for CO3D category: {category}")

    expected_shape = (num_points, 3)
    for path in files:
        points = np.load(path)
        if points.shape != expected_shape:
            raise RuntimeError(f"{path}: shape {points.shape} != {expected_shape}")
        if not np.isfinite(points).all():
            raise RuntimeError(f"{path}: contains NaN or Inf")
    return len(files)


def cleanup_targets(download_folder, category):
    root = Path(download_folder)
    targets = []
    category_lower = category.lower()

    def is_category_name(name):
        name = name.lower()
        return (
            name == category_lower
            or name.startswith(f"{category_lower}_")
            or name.startswith(f"{category_lower}.")
        )
    category_dir = root / category
    if category_dir.exists():
        targets.append(category_dir)

    if root.exists():
        for child in root.iterdir():
            name = child.name.lower()
            is_category_archive = (
                child.is_file()
                and is_category_name(name)
                and child.suffix.lower() in ARCHIVE_SUFFIXES
            )
            if is_category_archive:
                targets.append(child)
            if child.is_dir() and name == f"{category_lower}_in_progress":
                targets.append(child)

    in_progress = root / "_in_progress"
    if in_progress.is_dir():
        direct = in_progress / category
        if direct.exists():
            targets.append(direct)
        for child in in_progress.iterdir():
            if is_category_name(child.name):
                targets.append(child)

    deduped = []
    seen = set()
    for target in targets:
        resolved = str(Path(target).resolve())
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(Path(target))
    return deduped


def cleanup_raw(args, category):
    logger = get_logger("download_and_build_co3d_category")
    targets = cleanup_targets(args.download_folder, category)
    if not targets:
        logger.info(f"No raw cleanup targets: category={category}")
        return False

    cleaned = False
    for target in targets:
        if not ensure_under_root(target, args.download_folder):
            raise RuntimeError(f"Refusing to cleanup unsafe path: {target}")
        logger.info(f"Cleanup raw path: {target}")
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        cleaned = True
    return cleaned


def handle_failure(args, category, exc, summary):
    logger = get_logger("download_and_build_co3d_category")
    logger.exception(f"Failed category {category}: {exc}")
    summary["failed"].append(category)

    if args.no_cleanup_raw or args.download_only:
        logger.info(f"Keep raw after failure: {category}")
        summary["kept_raw"].append(category)

    if args.strict:
        raise RuntimeError(f"Stopping after failed CO3D category {category}") from exc


def process_one(args, category, downloader_options_cache, summary):
    logger = get_logger("download_and_build_co3d_category")

    if args.overwrite:
        remove_category_output(args.out_root, category)
    elif args.skip_existing and category_output_exists(args.out_root, category):
        logger.info(f"Skip existing output: {category}")
        summary["skipped"].append(category)
        if not args.no_cleanup_raw and not args.download_only:
            try:
                if cleanup_raw(args, category):
                    summary["cleaned"].append(category)
            except Exception as cleanup_exc:
                logger.exception(f"Cleanup failed for {category}: {cleanup_exc}")
                summary["kept_raw"].append(category)
        return

    try:
        if not args.process_only:
            if category_downloaded(args.download_folder, category):
                logger.info(f"Skip download: raw already present for {category}")
            else:
                downloader_options = get_downloader_options(args, downloader_options_cache)
                run_downloader(args, category, downloader_options)

        if args.download_only:
            logger.info(f"Download only: keep raw {category}")
            summary["succeeded"].append(category)
            summary["kept_raw"].append(category)
            return

        process_category(args, category)
        sample_count = validate_category_output(args.out_root, category, args.num_points)
        logger.info(f"Validated category {category}: {sample_count} samples")
        summary["succeeded"].append(category)
    except Exception as exc:
        handle_failure(args, category, exc, summary)
    finally:
        # Keep peak disk usage bounded to one category. Cleanup is also
        # attempted after failures before the next category is downloaded.
        if not args.no_cleanup_raw and not args.download_only:
            try:
                if cleanup_raw(args, category):
                    summary["cleaned"].append(category)
            except Exception as cleanup_exc:
                logger.exception(f"Cleanup failed for {category}: {cleanup_exc}")
                summary["kept_raw"].append(category)
        elif args.no_cleanup_raw and category not in summary["kept_raw"]:
            logger.info(f"Keep raw: {category}")
            summary["kept_raw"].append(category)


def log_summary(args, summary):
    logger = get_logger("download_and_build_co3d_category")
    logger.info("Summary")
    logger.info(f"  succeeded: {summary['succeeded']}")
    logger.info(f"  skipped: {summary['skipped']}")
    logger.info(f"  failed: {summary['failed']}")
    logger.info(f"  cleaned raw: {summary['cleaned']}")
    logger.info(f"  kept raw: {summary['kept_raw']}")
    logger.info(f"  output root: {Path(args.out_root) / 'CO3D'}")


def run(args):
    logger = get_logger(
        "download_and_build_co3d_category",
        log_file=os.path.join(args.out_root, "download_and_build_co3d_category.log"),
    )
    if args.download_only and args.process_only:
        raise ValueError("--download_only and --process_only cannot be used together")

    Path(args.download_folder).mkdir(parents=True, exist_ok=True)

    categories = resolve_category_targets(args)
    logger.info(f"Targets: {', '.join(categories)}")

    downloader_options_cache = {"value": None}
    summary = {
        "succeeded": [],
        "skipped": [],
        "failed": [],
        "cleaned": [],
        "kept_raw": [],
    }
    for category in categories:
        process_one(args, category, downloader_options_cache, summary)

    if not args.download_only:
        rebuild_split_manifest_from_outputs(
            args.out_root,
            seed=args.seed,
            train_ratio=0.8,
            report_path=Path(args.out_root) / "co3d_manifest_rebuild_report.json",
            validate_num_points=args.num_points,
            logger=logger,
        )

    log_summary(args, summary)
    if summary["failed"]:
        logger.warning("Some categories failed. Re-run with --strict to stop on first failure.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--co3d_repo", default="external/co3d")
    parser.add_argument("--download_folder", default="raw_datasets/co3d")
    parser.add_argument("--out_root", default="outputs")
    parser.add_argument("--category", default=None)
    parser.add_argument("--categories", default=None)
    parser.add_argument("--all_categories", action="store_true")
    parser.add_argument("--cross_classes_only", action="store_true")
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--download_only", action="store_true")
    parser.add_argument("--process_only", action="store_true")
    parser.add_argument("--cleanup_raw", dest="no_cleanup_raw", action="store_false", default=False)
    parser.add_argument("--no_cleanup_raw", dest="no_cleanup_raw", action="store_true")
    parser.add_argument(
        "--force_cleanup_on_error",
        action="store_true",
        help="Deprecated: cleanup after errors is now the default unless --no_cleanup_raw is used.",
    )
    parser.add_argument("--single_sequence_subset", action="store_true", default=False)
    parser.add_argument("--no_single_sequence_subset", dest="single_sequence_subset", action="store_false")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--clear_archives_after_unpacking", action="store_true")
    parser.add_argument("--checksum_check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
