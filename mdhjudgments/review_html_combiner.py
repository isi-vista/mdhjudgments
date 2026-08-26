"""Shared helpers for combining dual-review HTML reports."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
import html
from pathlib import Path
import sys

from bs4 import BeautifulSoup, FeatureNotFound
from bs4.element import Tag


@dataclass(frozen=True)
class ReviewConfig:
    """Presentation and parsing settings for one dual-review HTML combiner."""

    cli_description: str
    review_heading: str
    dangling_review_description: str
    title: str
    summary_count_label: str
    output_count_label: str
    layout_help: str
    group_class: str
    pane_class: str
    label_class: str
    primary_label: str
    secondary_label: str


@dataclass(frozen=True)
class ReviewSection:
    """The parsed tables for one reviewed excerpt section."""

    response_id: str
    section_id: str
    question_html: str
    section_html: str
    comments_html: str
    review_html: str

    @property
    def adjudication_html(self) -> str:
        """Return review HTML using the old adjudication field name."""
        return self.review_html

    @property
    def fact_check_html(self) -> str:
        """Return review HTML using the old fact-checking field name."""
        return self.review_html


def soup_from_text(text: str) -> BeautifulSoup:
    """Parse HTML with lxml when available, otherwise use the built-in parser."""
    try:
        return BeautifulSoup(text, "lxml")
    except FeatureNotFound:
        return BeautifulSoup(text, "html.parser")


def _table_heading(table: Tag) -> str:
    heading = table.find("th", class_="spanning")
    if not heading:
        return ""
    return heading.get_text("\n", strip=True)


def _table_value(table: Tag | None, label: str) -> str:
    if table is None:
        return ""

    for row in table.find_all("tr"):
        cells = row.find_all("td", recursive=False)
        if len(cells) >= 2 and cells[0].get_text(" ", strip=True) == label:
            return cells[1].get_text(" ", strip=True)
    return ""


def _has_class(tag: Tag, class_name: str) -> bool:
    return class_name in (tag.get("class") or [])


def extract_sections(path: Path, config: ReviewConfig) -> list[ReviewSection]:
    """Extract reviewed sections from one report, preserving report order."""
    soup = soup_from_text(path.read_text(encoding="utf-8"))
    sections: list[ReviewSection] = []

    for box_index, box in enumerate(soup.find_all("div", class_="response-box"), start=1):
        question_table = box.find("table", class_="question-table", recursive=False)
        if question_table is None:
            print(f"{path}: response box {box_index} has no question table", file=sys.stderr)
            continue
        response_id = _table_value(question_table, "Response ID")

        current_section: Tag | None = None
        current_comments: Tag | None = None

        for table in box.find_all("table", recursive=False):
            heading = _table_heading(table)

            if _has_class(table, "question-table"):
                continue

            if _has_class(table, "section-table") and heading.startswith("SECTION:"):
                current_section = table
                current_comments = None
                continue

            if _has_class(table, "comments-table"):
                current_comments = table
                continue

            if _has_class(table, "section-table") and heading == config.review_heading:
                if current_section is None:
                    print(
                        (
                            f"{path}: response box {box_index} has "
                            f"{config.dangling_review_description} with no section"
                        ),
                        file=sys.stderr,
                    )
                    continue

                section_id = _table_value(current_section, "Section ID")
                if not section_id:
                    print(
                        f"{path}: response box {box_index} has a section with no Section ID",
                        file=sys.stderr,
                    )
                    continue

                sections.append(
                    ReviewSection(
                        response_id=response_id,
                        section_id=section_id,
                        question_html=str(question_table),
                        section_html=str(current_section),
                        comments_html=str(current_comments) if current_comments else "",
                        review_html=str(table),
                    )
                )
                current_section = None
                current_comments = None

    return sections


def index_sections(sections: Iterable[ReviewSection], source_name: str) -> dict[str, ReviewSection]:
    """Index sections by Section ID, rejecting duplicate IDs."""
    by_id: dict[str, ReviewSection] = {}
    for section in sections:
        if section.section_id in by_id:
            raise ValueError(f"{source_name}: duplicate Section ID {section.section_id}")
        by_id[section.section_id] = section
    return by_id


def _normalize_html_fragment(fragment: str) -> str:
    return " ".join(soup_from_text(fragment).get_text(" ", strip=True).split())


def render_review(label: str, source: Path, review_html: str, config: ReviewConfig) -> str:
    """Render one review pane for the combined report."""
    label_escaped = html.escape(label)
    source_escaped = html.escape(source.name)
    return f"""
        <section class="{config.pane_class}" aria-label="{label_escaped}">
          <div class="{config.label_class}">
            <span>{label_escaped}</span>
            <code>{source_escaped}</code>
          </div>
          {review_html}
        </section>
    """


def build_output_html(
    primary_path: Path,
    secondary_path: Path,
    primary_sections: list[ReviewSection],
    secondary_by_id: dict[str, ReviewSection],
    layout: str,
    config: ReviewConfig,
) -> tuple[str, int, int]:
    """Build combined review HTML and return it with section/comment counts."""
    dual_sections = [
        section for section in primary_sections if section.section_id in secondary_by_id
    ]
    differing_comments = 0

    layout_class = (
        f"{config.group_class}--stacked"
        if layout == "stacked"
        else f"{config.group_class}--side-by-side"
    )

    parts: list[str] = [
        f"""<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <title>{html.escape(config.title)}</title>
    <style>
      body {{
        margin: 24px;
        color: #1f2933;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        line-height: 1.35;
        background: #f7f9fc;
      }}

      .summary {{
        margin: 0 0 18px 0;
        padding: 12px 14px;
        border: 1px solid #b8c4d6;
        background: #ffffff;
      }}

      .summary h1 {{
        margin: 0 0 8px 0;
        font-size: 1.25rem;
      }}

      .summary p {{
        margin: 4px 0;
      }}

      .response-box {{
        border: 2px solid #bbb;
        border-radius: 6px;
        padding: 10px 12px;
        margin: 18px 0 24px 0;
        background: #fcfcff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
      }}

      .dual-section {{
        border-top: 3px solid #d7dee8;
        margin-top: 18px;
        padding-top: 12px;
      }}

      .dual-section:first-of-type {{
        border-top: 0;
        margin-top: 0;
        padding-top: 0;
      }}

      table.question-table, table.section-table, table.comments-table {{
        border-collapse: collapse;
        border: 1px solid #666;
        width: 100%;
      }}

      table.question-table th, table.question-table td,
      table.section-table th, table.section-table td,
      table.comments-table th, table.comments-table td {{
        border: 1px solid #666;
        padding: 6px 8px;
        text-align: left;
        vertical-align: top;
        word-wrap: break-word;
        white-space: normal;
      }}

      table.question-table {{
        margin: 6px 0 10px 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
        border: 2px solid #2c7be5;
      }}

      table.question-table th.spanning,
      table.section-table th.spanning,
      table.comments-table th.spanning {{
        text-align: left;
        background: #f0f0f0;
        font-weight: bold;
        padding: 8px;
      }}

      table.question-table th.spanning {{
        background: #e9f2ff;
        border-bottom: 2px solid #2c7be5;
        font-size: 1.05rem;
        font-weight: 700;
        padding: 10px;
      }}

      table.section-table, table.comments-table {{
        margin: 10px 0 20px 0;
      }}

      td.label {{
        width: 10em;
      }}

      td.gloss {{
        width: 30em;
      }}

      .{config.group_class} {{
        display: grid;
        gap: 16px;
        align-items: start;
        margin: 12px 0 20px 0;
      }}

      .{config.group_class}--side-by-side {{
        grid-template-columns: repeat(2, minmax(420px, 1fr));
      }}

      .{config.group_class}--stacked {{
        grid-template-columns: 1fr;
      }}

      .{config.pane_class} {{
        min-width: 0;
        overflow-x: auto;
      }}

      .{config.label_class} {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px 12px;
        align-items: baseline;
        margin: 0 0 6px 0;
        font-weight: 700;
      }}

      .{config.label_class} code {{
        font-size: 0.85rem;
        font-weight: 400;
      }}

      .{config.pane_class} table.section-table {{
        margin-top: 0;
      }}

      @media (max-width: 1100px) {{
        .{config.group_class}--side-by-side {{
          grid-template-columns: 1fr;
        }}

        body {{
          margin: 12px;
        }}
      }}
    </style>
  </head>
  <body>
""",
        f"""    <section class="summary">
      <h1>{html.escape(config.title)}</h1>
      <p>Showing only sections present in both input files.</p>
      <p><strong>Primary:</strong> {html.escape(primary_path.name)}</p>
      <p><strong>Secondary:</strong> {html.escape(secondary_path.name)}</p>
      <p><strong>{html.escape(config.summary_count_label)}:</strong> {len(dual_sections)}</p>
    </section>
""",
    ]

    current_response_id: str | None = None
    response_box_open = False

    for primary in dual_sections:
        secondary = secondary_by_id[primary.section_id]
        if _normalize_html_fragment(primary.comments_html) != _normalize_html_fragment(
            secondary.comments_html
        ):
            differing_comments += 1

        if primary.response_id != current_response_id:
            if response_box_open:
                parts.append("    </div>\n")
            parts.append('    <div class="response-box">\n')
            parts.append(primary.question_html)
            parts.append("\n")
            current_response_id = primary.response_id
            response_box_open = True

        parts.append('      <section class="dual-section">\n')
        parts.append(primary.section_html)
        parts.append("\n")
        if primary.comments_html:
            parts.append(primary.comments_html)
            parts.append("\n")
        parts.append(f'        <div class="{config.group_class} {layout_class}">\n')
        parts.append(render_review(config.primary_label, primary_path, primary.review_html, config))
        parts.append(
            render_review(config.secondary_label, secondary_path, secondary.review_html, config)
        )
        parts.append("        </div>\n")
        parts.append("      </section>\n")

    if response_box_open:
        parts.append("    </div>\n")

    parts.append(
        """  </body>
</html>
"""
    )
    return "".join(parts), len(dual_sections), differing_comments


def parse_args(config: ReviewConfig, argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for a configured combiner."""
    parser = argparse.ArgumentParser(description=config.cli_description)
    parser.add_argument("primary", nargs="?", type=Path)
    parser.add_argument("secondary", nargs="?", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "--layout",
        choices=("side-by-side", "stacked"),
        default="side-by-side",
        help=config.layout_help,
    )
    return parser.parse_args(argv)


def run_combiner(config: ReviewConfig, argv: list[str] | None = None) -> int:
    """Run a configured dual-review HTML combiner command."""
    args = parse_args(config, argv)
    if args.primary is None or args.secondary is None or args.output is None:
        print("primary, secondary, and --output are required", file=sys.stderr)
        return 2

    primary_path: Path = args.primary
    secondary_path: Path = args.secondary
    output_path: Path = args.output

    for path in (primary_path, secondary_path):
        if not path.exists():
            print(f"Input file not found: {path}", file=sys.stderr)
            return 2

    primary_sections = extract_sections(primary_path, config)
    secondary_sections = extract_sections(secondary_path, config)
    secondary_by_id = index_sections(secondary_sections, secondary_path.name)

    output_html, dual_count, differing_comments = build_output_html(
        primary_path=primary_path,
        secondary_path=secondary_path,
        primary_sections=primary_sections,
        secondary_by_id=secondary_by_id,
        layout=args.layout,
        config=config,
    )
    output_path.write_text(output_html, encoding="utf-8")

    print(f"Wrote {output_path}")
    print(f"Primary sections: {len(primary_sections)}")
    print(f"Secondary sections: {len(secondary_sections)}")
    print(f"{config.output_count_label}: {dual_count}")
    if differing_comments:
        print(
            f"Warning: comments table text differed for {differing_comments} dual sections; "
            "the output shows the primary file's comments.",
            file=sys.stderr,
        )

    return 0
