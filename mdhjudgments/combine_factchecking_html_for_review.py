#!/usr/bin/env python3
"""Combine two fact-checking HTML exports into a dual-annotated view.

The inputs are expected to be static HTML reports whose response boxes contain:

1. a question table,
2. one or more section tables,
3. a comments table after each section,
4. a fact-checking table after each comments table.

The output keeps only sections that appear in both inputs, keyed by Section ID.
For each matching section, it shows the question/section/comments once, followed
by the two fact-checking tables.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from mdhjudgments.review_html_combiner import (
    ReviewConfig,
    ReviewSection,
    build_output_html,
    extract_sections as _extract_configured_sections,
    index_sections,
    parse_args,
    render_review,
    run_combiner,
)

FACT_CHECKING_CONFIG = ReviewConfig(
    cli_description=(
        "Combine two fact-checking HTML reports and keep only sections present in both."
    ),
    review_heading="Fact-Checking",
    dangling_review_description="fact-checking",
    title="Dual Fact-Checking Summary",
    summary_count_label="Dual-annotated sections",
    output_count_label="Dual-annotated sections",
    layout_help="How to display the two fact-checking tables. Defaults to side-by-side.",
    group_class="fact-checks",
    pane_class="fact-checker-pane",
    label_class="fact-checker-label",
    primary_label="Fact-Checker 1",
    secondary_label="Fact-Checker 2",
)

AnnotatedSection = ReviewSection


def extract_sections(path: Path) -> list[AnnotatedSection]:
    """Extract fact-checked sections from one report, preserving report order."""
    return _extract_configured_sections(path, FACT_CHECKING_CONFIG)


def _index_sections(
    sections: Iterable[AnnotatedSection], source_name: str
) -> dict[str, AnnotatedSection]:
    return index_sections(sections, source_name)


def _render_fact_check(label: str, source: Path, fact_check_html: str) -> str:
    return render_review(label, source, fact_check_html, FACT_CHECKING_CONFIG)


def _build_output_html(
    primary_path: Path,
    secondary_path: Path,
    primary_sections: list[AnnotatedSection],
    secondary_by_id: dict[str, AnnotatedSection],
    layout: str,
) -> tuple[str, int, int]:
    return build_output_html(
        primary_path=primary_path,
        secondary_path=secondary_path,
        primary_sections=primary_sections,
        secondary_by_id=secondary_by_id,
        layout=layout,
        config=FACT_CHECKING_CONFIG,
    )


def _parse_args() -> argparse.Namespace:
    return parse_args(FACT_CHECKING_CONFIG)


def _main() -> int:
    return run_combiner(FACT_CHECKING_CONFIG)


if __name__ == "__main__":
    raise SystemExit(_main())
