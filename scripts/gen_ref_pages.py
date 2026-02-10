"""Generate the code reference pages and navigation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

DEFAULT_FOLDERS = ("jarvis", "models", "api", "core", "integrations")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folders",
        nargs="+",
        default=list(DEFAULT_FOLDERS),
        help="Source folders to scan for Python modules (default: %(default)s).",
    )
    parser.add_argument(
        "--reference-root",
        default="reference",
        help="Documentation output root used by mkdocs-gen-files (default: %(default)s).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N files while generating references (default: %(default)s).",
    )
    return parser.parse_args(argv)


def generate_reference_pages(
    src_folders: Sequence[str],
    reference_root: str = "reference",
    progress_every: int = 50,
) -> None:
    """Generate reference markdown pages and SUMMARY navigation."""
    import mkdocs_gen_files

    nav = mkdocs_gen_files.Nav()
    total_docs = 0

    for folder in src_folders:
        src = Path(folder)
        if not src.exists():
            print(f"Skipping missing source folder: {src}", flush=True)
            continue

        paths = sorted(src.rglob("*.py"))
        if len(paths) > 10:
            print(f"Processing {len(paths)} modules under {src}...", flush=True)

        for idx, path in enumerate(paths, 1):
            module_path = path.with_suffix("")
            doc_path = path.with_suffix(".md")
            full_doc_path = Path(reference_root, doc_path)

            parts = tuple(module_path.parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
                doc_path = doc_path.with_name("index.md")
                full_doc_path = full_doc_path.with_name("index.md")
            elif parts[-1] == "__main__":
                continue

            nav[parts] = doc_path.as_posix()
            try:
                with mkdocs_gen_files.open(full_doc_path, "w") as fd:
                    ident = ".".join(parts)
                    fd.write(f"::: {ident}")
            except OSError as exc:
                print(f"Error writing reference page '{full_doc_path}': {exc}", file=sys.stderr, flush=True)
                raise SystemExit(1) from exc

            mkdocs_gen_files.set_edit_path(full_doc_path, path)
            total_docs += 1

            if len(paths) > 10 and progress_every > 0 and (idx % progress_every == 0 or idx == len(paths)):
                print(f"  {src}: {idx}/{len(paths)} modules processed", flush=True)

    summary_path = Path(reference_root, "SUMMARY.md")
    try:
        with mkdocs_gen_files.open(summary_path, "w") as nav_file:
            nav_file.writelines(nav.build_literate_nav())
    except OSError as exc:
        print(f"Error writing summary page '{summary_path}': {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc

    print(f"Generated {total_docs} reference pages under '{reference_root}/'.", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    generate_reference_pages(args.folders, args.reference_root, args.progress_every)


if __name__ == "__main__":
    main()
