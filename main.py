import argparse
import sys
from pathlib import Path

from engine import SortConfig, sort_images


def looks_like_placeholder(path: Path) -> bool:
    raw = str(path).lower()
    return any(token in raw for token in ("c:\\path\\to\\", "/path/to/"))


def validate_cli_paths(
    scan_folder: Path,
    output_folder: Path,
    reference_folder: Path | None,
    auto_cluster: bool,
) -> None:
    if looks_like_placeholder(scan_folder) or looks_like_placeholder(output_folder):
        raise ValueError(
            "Replace the example paths with real folders. Do not use literals like C:\\path\\to\\photos."
        )
    if reference_folder and looks_like_placeholder(reference_folder):
        raise ValueError(
            "Replace --reference-folder with a real folder path, or omit it for auto grouping."
        )
    if not scan_folder.exists():
        raise FileNotFoundError(f"Scan folder does not exist: {scan_folder}")
    if not scan_folder.is_dir():
        raise NotADirectoryError(f"Scan folder is not a directory: {scan_folder}")
    if not auto_cluster and reference_folder and not reference_folder.exists():
        raise FileNotFoundError(f"Reference folder does not exist: {reference_folder}")
    if (
        reference_folder
        and reference_folder.exists()
        and not reference_folder.is_dir()
    ):
        raise NotADirectoryError(f"Reference folder is not a directory: {reference_folder}")


def cli_progress(phase: str, current: int, total: int, message: str, detail: str = "") -> None:
    if phase == "done":
        print(message)
        if detail:
            print(detail)
        return
    if total > 1:
        prefix = f"[{current}/{total}]"
    else:
        prefix = ""
    line = f"{prefix} {message}".strip()
    if detail:
        line = f"{line} — {detail}"
    print(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sort photos by face recognition into similar-face groups."
    )
    parser.add_argument("scan_folder", help="Folder containing photos to scan.")
    parser.add_argument("output_folder", help="Folder where organized images will be copied.")
    parser.add_argument(
        "--reference-folder",
        default=None,
        help=(
            "Optional folder with one subfolder per person for named matching. "
            "If omitted, similar faces are grouped automatically."
        ),
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional CSV report path. Defaults to <output_folder>/face_report.csv.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers for face detection (default: CPU count, max 8).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scan_folder = Path(args.scan_folder).expanduser().resolve()
    output_folder = Path(args.output_folder).expanduser().resolve()
    reference_folder = (
        Path(args.reference_folder).expanduser().resolve()
        if args.reference_folder
        else None
    )
    auto_cluster = reference_folder is None

    validate_cli_paths(scan_folder, output_folder, reference_folder, auto_cluster)

    config = SortConfig(
        scan_folder=scan_folder,
        output_folder=output_folder,
        reference_folder=reference_folder,
        report_path=Path(args.report_path).expanduser().resolve() if args.report_path else None,
        auto_cluster=auto_cluster,
        workers=args.workers or SortConfig.workers,
    )

    result = sort_images(config, progress=cli_progress)
    print(f"Processed {result.image_count} images ({result.face_count} faces)")
    print(f"Created {result.group_count} face groups in {output_folder}")
    print(f"Wrote match report to {result.report_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(
            'Example: .\\.venv\\Scripts\\python.exe main.py "C:\\Users\\husse\\Pictures\\photos" '
            '"C:\\Users\\husse\\Pictures\\sorted"',
            file=sys.stderr,
        )
        sys.exit(2)
