"""Render first-pass annotations and all section reviews in one HTML report."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
import datetime
from enum import Enum
from html import escape
import json
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from mdhjudgments.file_utils import read_jsonl
from mdhjudgments.model import (
    EnhancedResponse,
    FactCheckingCommentCategory,
    InformationAccuracyAnnotationModel,
    SectionAdjudicationModel,
    SectionFactCheckingModel,
)

Review = TypeVar("Review", SectionAdjudicationModel, SectionFactCheckingModel)


@dataclass(frozen=True)
class SourcedReview(Generic[Review]):
    """A section review together with the input file that supplied it."""

    source: Path
    review: Review


def _format_value(value: Any) -> str:
    """Format a model value for display without losing nested data."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return json.dumps(value.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    if isinstance(value, dict):
        return json.dumps(
            {
                str(key): item.model_dump(mode="json") if isinstance(item, BaseModel) else item
                for key, item in value.items()
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    if isinstance(value, list | tuple | set):
        return "\n".join(_format_value(item) for item in value)
    return str(value)


def _cell(value: Any, *, tag: str = "td", css_class: str | None = None) -> str:
    """Render one escaped HTML table cell."""
    class_attribute = f' class="{css_class}"' if css_class else ""
    return f"<{tag}{class_attribute}>{escape(_format_value(value))}</{tag}>"


def _row(values: Iterable[Any], *, header: bool = False) -> str:
    """Render an HTML table row."""
    tag = "th" if header else "td"
    return "<tr>" + "".join(_cell(value, tag=tag) for value in values) + "</tr>"


def _key_value_table(title: str, values: Iterable[tuple[str, Any]]) -> str:
    """Render a titled two-column table."""
    rows = "".join(
        f"<tr>{_cell(label, css_class='label')}{_cell(value)}</tr>" for label, value in values
    )
    return (
        '<table class="data-table review-table">'
        f'<tr><th class="spanning" colspan="2">{escape(title)}</th></tr>{rows}</table>'
    )


def _read_responses(path: Path) -> list[EnhancedResponse]:
    """Load an EnhancedResponse JSONL file."""
    return [EnhancedResponse.model_validate(row) for row in read_jsonl(path)]


def _collate_adjudications(
    paths: list[Path],
) -> dict[str, list[SourcedReview[SectionAdjudicationModel]]]:
    """Collect medical-expert adjudications by section, preserving input order."""
    result: dict[str, list[SourcedReview[SectionAdjudicationModel]]] = {}
    for path in paths:
        seen: set[str] = set()
        for response in _read_responses(path):
            if response.annotations is None:
                continue
            for section in response.annotations.sections_with_annotations:
                if section.adjudication is None:
                    continue
                if section.id in seen:
                    raise ValueError(f"{path}: duplicate adjudication for section {section.id!r}")
                seen.add(section.id)
                result.setdefault(section.id, []).append(
                    SourcedReview(source=path, review=section.adjudication)
                )
    return result


def _collate_fact_checkings(
    paths: list[Path],
) -> dict[str, list[SourcedReview[SectionFactCheckingModel]]]:
    """Collect fact-checker reviews by section, preserving input order."""
    result: dict[str, list[SourcedReview[SectionFactCheckingModel]]] = {}
    for path in paths:
        seen: set[str] = set()
        for response in _read_responses(path):
            if response.annotations is None:
                continue
            for section in response.annotations.sections_with_annotations:
                if section.fact_checking is None:
                    continue
                if section.id in seen:
                    raise ValueError(f"{path}: duplicate fact-checking for section {section.id!r}")
                seen.add(section.id)
                result.setdefault(section.id, []).append(
                    SourcedReview(source=path, review=section.fact_checking)
                )
    return result


def _render_answer_annotations(response: EnhancedResponse) -> str:
    """Render every response-level annotation."""
    assert response.annotations is not None
    annotations = response.annotations.answer_annotations
    rows = "".join(
        _row(
            [
                annotation.id,
                annotation.annotator_id,
                annotation.timestamp,
                annotation.dimensions.accuracy if annotation.dimensions is not None else None,
                annotation.sources,
            ]
        )
        for annotation in annotations
    )
    return (
        '<table class="data-table annotation-table">'
        '<tr><th class="spanning" colspan="5">Per-Answer Annotations</th></tr>'
        + _row(["ID", "Annotator ID", "Timestamp", "Accuracy", "Sources"], header=True)
        + rows
        + ("" if annotations else '<tr><td colspan="5" class="empty">None</td></tr>')
        + "</table>"
    )


def _render_section_annotations(
    section_id: str, annotations: list[InformationAccuracyAnnotationModel]
) -> str:
    """Render every first-pass annotation for a section."""
    rows = "".join(
        _row(
            [
                annotation.id,
                annotation.annotator_id,
                annotation.timestamp,
                annotation.accuracy_type,
                annotation.element_correctness.certainty,
                annotation.element_correctness.risk,
                annotation.element_correctness.urgency,
                annotation.comment,
            ]
        )
        for annotation in annotations
    )
    return (
        '<table class="data-table annotation-table">'
        '<tr><th class="spanning" colspan="8">First-Pass Section Annotations</th></tr>'
        + _row(
            [
                "ID",
                "Annotator ID",
                "Timestamp",
                "Accuracy Type",
                "Certainty",
                "Risk",
                "Urgency",
                "Comment",
            ],
            header=True,
        )
        + rows
        + (
            ""
            if annotations
            else f'<tr><td colspan="8" class="empty">No annotations for {escape(section_id)}</td></tr>'
        )
        + "</table>"
    )


def _render_adjudication(
    review: SourcedReview[SectionAdjudicationModel],
    index: int,
    annotations_by_id: dict[str, InformationAccuracyAnnotationModel],
) -> str:
    """Render one medical-expert adjudication pane."""
    adjudication = review.review
    value_rows = [
        ("Adjudicator ID", adjudication.annotator_id),
        ("Timestamp", adjudication.timestamp),
        ("Aggregate Judgment", adjudication.aggregate_judgment),
        ("Q1", adjudication.q1_response),
    ]
    value_rows_html = "".join(
        f"<tr>{_cell(label, css_class='label')}<td colspan=\"2\">"
        f"{escape(_format_value(value))}</td></tr>"
        for label, value in value_rows
    )
    q2_rows = list(adjudication.q2_category_response.items())
    q2_html = (
        f'<tr><td class="label" rowspan="{max(1, len(q2_rows)) + 1}">Q2 Categories</td>'
        "<th>Comment text</th><th>Categories applied</th></tr>"
        + "".join(
            _row(
                [
                    (
                        annotations_by_id[comment_id].comment
                        if comment_id in annotations_by_id
                        else "Annotation text unavailable"
                    ),
                    [str(category) for category in categories],
                ]
            )
            for comment_id, categories in q2_rows
        )
        + ("" if q2_rows else '<tr><td colspan="2" class="empty">No categories supplied</td></tr>')
    )
    trailing_rows_html = "".join(
        f"<tr>{_cell(label, css_class='label')}<td colspan=\"2\">"
        f"{escape(_format_value(value))}</td></tr>"
        for label, value in [
            ("Q3", adjudication.q3_response),
            ("Comment", adjudication.comment),
            ("Saw Annotation IDs", adjudication.saw_annotation_ids),
        ]
    )
    table = (
        '<table class="data-table review-table">'
        '<tr><th class="spanning" colspan="3">Medical-Expert Adjudication</th></tr>'
        f"{value_rows_html}{q2_html}{trailing_rows_html}</table>"
    )
    return (
        '<section class="review-pane">'
        f'<div class="review-label">Medical Expert {index} <code>{escape(review.source.name)}</code></div>'
        f"{table}</section>"
    )


def _fact_checking_categories(categories: list[FactCheckingCommentCategory]) -> list[str]:
    """Format fact-checking comment categories, including Other free text."""
    return [
        str(category.simple_category)
        + (f" ({category.other_text})" if category.other_text is not None else "")
        for category in categories
    ]


def _render_fact_checking(review: SourcedReview[SectionFactCheckingModel], index: int) -> str:
    """Render one fact-checker review pane."""
    fact_checking = review.review
    table = _key_value_table(
        "Fact-Checking",
        [
            ("Fact-Checker ID", fact_checking.annotator_id),
            ("Timestamp", fact_checking.timestamp),
            ("Aggregate Judgment", fact_checking.aggregate_judgment),
            ("Q1", fact_checking.q1_response),
            ("Q2 Categories", _fact_checking_categories(fact_checking.q2_category_response)),
            ("Q3 Choice", fact_checking.q3_choice),
            ("Q3 URL", fact_checking.q3_url),
            ("Q4 Evidence Sentences", fact_checking.q4_excerpt),
            ("Q5a Choice", fact_checking.q5a_choice),
            ("Q5b Choice", fact_checking.q5b_choice),
            ("Q5c Choice", fact_checking.q5c_choice),
            ("Q6 Comment", fact_checking.q6_comment),
            ("Saw Annotation IDs", fact_checking.saw_annotation_ids),
        ],
    )
    return (
        '<section class="review-pane">'
        f'<div class="review-label">Fact-Checker {index} <code>{escape(review.source.name)}</code></div>'
        f"{table}</section>"
    )


def _render_review_group(title: str, panes: list[str]) -> str:
    """Render reviews in a responsive two-column group."""
    if not panes:
        return ""
    return (
        '<section class="review-group">'
        f"<h3>{escape(title)}</h3>"
        f'<div class="review-grid">{"".join(panes)}</div></section>'
    )


def build_html(
    responses: list[EnhancedResponse],
    adjudications: dict[str, list[SourcedReview[SectionAdjudicationModel]]],
    fact_checkings: dict[str, list[SourcedReview[SectionFactCheckingModel]]],
) -> str:
    """Build a combined HTML report from canonical annotations and section reviews."""
    body: list[str] = []
    section_count = 0
    for response in responses:
        if response.annotations is None:
            continue
        response_parts = [
            '<article class="response-box">',
            '<table class="data-table question-table">',
            f'<tr><th class="spanning" colspan="2">QUESTION:<br>{escape(response.question)}</th></tr>',
            f"<tr>{_cell('Response ID', css_class='label')}{_cell(response.id)}</tr>",
            f"<tr>{_cell('Answer', css_class='label')}{_cell(response.answer)}</tr>",
            "</table>",
            _render_answer_annotations(response),
        ]
        for section in response.annotations.sections_with_annotations:
            section_count += 1
            annotations_by_id = {annotation.id: annotation for annotation in section.annotations}
            response_parts.extend(
                [
                    '<section class="section-box">',
                    '<table class="data-table section-table">',
                    f'<tr><th class="spanning" colspan="2">SECTION:<br>{escape(section.section)}</th></tr>',
                    f"<tr>{_cell('Section ID', css_class='label')}{_cell(section.id)}</tr>",
                    "</table>",
                    _render_section_annotations(section.id, section.annotations),
                    _render_review_group(
                        "Medical-Expert Adjudications",
                        [
                            _render_adjudication(
                                review,
                                index,
                                annotations_by_id,
                            )
                            for index, review in enumerate(
                                adjudications.get(section.id, []), start=1
                            )
                        ],
                    ),
                    _render_review_group(
                        "Fact-Checker Adjudications",
                        [
                            _render_fact_checking(review, index)
                            for index, review in enumerate(
                                fact_checkings.get(section.id, []), start=1
                            )
                        ],
                    ),
                    "</section>",
                ]
            )
        response_parts.append("</article>")
        body.append("".join(response_parts))

    css = """
      :root { color-scheme: light; }
      body { margin: 24px; color: #1f2933; background: #f7f9fc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.35; }
      .summary, .response-box { background: #fff; border: 1px solid #b8c4d6; border-radius: 6px; }
      .summary { padding: 12px 14px; margin-bottom: 18px; }
      .summary h1 { margin: 0 0 8px; font-size: 1.3rem; }
      .summary p { margin: 4px 0; }
      .response-box { padding: 12px; margin: 18px 0 28px; box-shadow: 0 1px 3px #00000012; }
      .section-box { border-top: 3px solid #d7dee8; margin-top: 22px; padding-top: 16px; }
      .data-table { width: 100%; border-collapse: collapse; margin: 10px 0 18px; table-layout: auto; }
      .data-table th, .data-table td { border: 1px solid #667; padding: 6px 8px;
        text-align: left; vertical-align: top; white-space: pre-wrap; overflow-wrap: anywhere; }
      .data-table th { background: #f0f0f0; }
      .question-table { border: 2px solid #2c7be5; }
      .question-table th.spanning { background: #e9f2ff; border-bottom: 2px solid #2c7be5; }
      th.spanning { font-size: 1.03rem; }
      td.label { width: 12em; font-weight: 600; }
      td.empty { color: #667085; font-style: italic; }
      .annotation-table { display: block; overflow-x: auto; }
      .review-group h3 { margin: 18px 0 8px; }
      .review-grid { display: grid; grid-template-columns: repeat(2, minmax(420px, 1fr));
        gap: 16px; align-items: start; }
      .review-pane { min-width: 0; overflow-x: auto; }
      .review-pane:only-child { grid-column: 1 / -1; }
      .review-label { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px;
        padding: 7px 9px; background: #e8edf5; font-weight: 700; }
      .review-label code { font-size: .82rem; font-weight: 400; }
      .review-table { margin-top: 0; }
      @media (max-width: 1100px) {
        body { margin: 12px; }
        .review-grid { grid-template-columns: 1fr; }
      }
    """
    summary = (
        '<section class="summary"><h1>Combined Annotation and Adjudication Summary</h1>'
        f"<p><strong>Responses:</strong> {len(body)}</p>"
        f"<p><strong>Sections:</strong> {section_count}</p></section>"
    )
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        "<title>Combined Annotation and Adjudication Summary</title>"
        f"<style>{css}</style></head><body>{summary}{''.join(body)}</body></html>"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-file",
        type=Path,
        required=True,
        help="JSONL file of EnhancedResponse first-pass annotation data.",
    )
    parser.add_argument(
        "--adjudication-files",
        type=Path,
        nargs="+",
        required=True,
        help="JSONL files with medical-expert section adjudications.",
    )
    parser.add_argument(
        "--factchecking-files",
        type=Path,
        nargs="+",
        required=True,
        help="JSONL files with fact-checker section adjudications.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output HTML path.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Run the combined visualization command."""
    args = _parse_args()
    input_paths = [
        args.annotation_file,
        *args.adjudication_files,
        *args.factchecking_files,
    ]
    missing = [path for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Input files do not exist or are not files: {missing!r}")

    responses = _read_responses(args.annotation_file)
    adjudications = _collate_adjudications(args.adjudication_files)
    fact_checkings = _collate_fact_checkings(args.factchecking_files)
    args.output.write_text(
        build_html(responses, adjudications, fact_checkings),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
