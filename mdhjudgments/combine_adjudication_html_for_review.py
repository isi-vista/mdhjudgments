#!/usr/bin/env python3
"""Combine two adjudication HTML exports into a dual-adjudicated view.

The inputs are expected to be static HTML reports whose response boxes contain:

1. a question table,
2. one or more section tables,
3. a comments table after each section,
4. an adjudication table after each comments table.

The output keeps only sections that appear in both inputs, keyed by Section ID.
For each matching section, it shows the question/section/comments once, followed
by the two adjudication tables.
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

ADJUDICATION_CONFIG = ReviewConfig(
    cli_description="Combine two adjudication HTML reports and keep only sections present in both.",
    review_heading="Adjudication",
    dangling_review_description="adjudication",
    title="Dual Adjudication Summary",
    summary_count_label="Dual-adjudicated sections",
    output_count_label="Dual-adjudicated sections",
    layout_help="How to display the two adjudication tables. Defaults to side-by-side.",
    group_class="adjudications",
    pane_class="adjudicator-pane",
    label_class="adjudicator-label",
    primary_label="Adjudicator 1",
    secondary_label="Adjudicator 2",
)

AdjudicatedSection = ReviewSection


def extract_sections(path: Path) -> list[AdjudicatedSection]:
    """Extract adjudicated sections from one report, preserving report order."""
    return _extract_configured_sections(path, ADJUDICATION_CONFIG)


def _index_sections(
    sections: Iterable[AdjudicatedSection], source_name: str
) -> dict[str, AdjudicatedSection]:
    return index_sections(sections, source_name)


def _render_adjudication(label: str, source: Path, adjudication_html: str) -> str:
    return render_review(label, source, adjudication_html, ADJUDICATION_CONFIG)


def _build_output_html(
    primary_path: Path,
    secondary_path: Path,
    primary_sections: list[AdjudicatedSection],
    secondary_by_id: dict[str, AdjudicatedSection],
    layout: str,
) -> tuple[str, int, int]:
    return build_output_html(
        primary_path=primary_path,
        secondary_path=secondary_path,
        primary_sections=primary_sections,
        secondary_by_id=secondary_by_id,
        layout=layout,
        config=ADJUDICATION_CONFIG,
    )


def _parse_args() -> argparse.Namespace:
    return parse_args(ADJUDICATION_CONFIG)


def _main() -> int:
    return run_combiner(ADJUDICATION_CONFIG)


if __name__ == "__main__":
    raise SystemExit(_main())
