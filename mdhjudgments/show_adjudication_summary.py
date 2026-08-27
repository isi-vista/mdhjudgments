"""Script to render adjudication outcomes given adjudicated annotation data.

Takes a file of EnhancedResponses and renders all (and only) the sections that went through adjudication.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
from html import escape
import logging
from pathlib import Path

from dominate import document
from dominate.tags import div, meta, style, table, td, th, tr

from mdhjudgments.file_utils import read_jsonl
from mdhjudgments.model import (
    EnhancedResponse,
    SectionResponseAnnotation,
)

logger = logging.getLogger(__name__)


@dataclass
class CommentEntry:
    """Represents a single comment, ready to be rendered as HTML."""

    comment_id: str
    annotator_id: str
    accuracy_type: str
    certainty: str
    risk: str
    urgency: str
    categories: str
    comment: str


@dataclass
class SectionEntry:
    """Represents a section's annotation and adjudication outcome, ready to be rendered as HTML."""

    section_id: str
    section_text: str
    aggregate_judgment: str
    q1_response: str
    q3_response: str
    adjudicator_id: str
    adjudicator_comment: str
    answer_key_impact: str | None
    comment_rows: list[CommentEntry]


@dataclass
class ResponseEntry:
    """Represents a question-sections pair ready to be rendered as HTML."""

    question: str
    response_id: str
    sections: list[SectionEntry]


def format_text(text: str) -> str:
    """Format text so it can be included directly in rendered HTML.

    This means escaping HTML special characters and adding <br> before literal newlines.
    """
    return escape(text).replace("\n", "<br>\n")


def build_comment_rows(
    section: SectionResponseAnnotation,
) -> list[CommentEntry]:
    """Build comment entries ready to be rendered as HTML."""
    assert section.adjudication is not None

    annotations_by_id = {annotation.id: annotation for annotation in section.annotations}
    annotated_but_not_adjudicated = set(annotations_by_id.keys()) - set(
        section.adjudication.saw_annotation_ids
    )
    if annotated_but_not_adjudicated:
        logger.debug(
            "Found %d comments annotated but not adjudicated --- IDs: %s",
            len(annotated_but_not_adjudicated),
            sorted(annotated_but_not_adjudicated),
        )

    rows: list[CommentEntry] = []

    for comment_id in section.adjudication.saw_annotation_ids:
        annotation = annotations_by_id[comment_id]
        categories = section.adjudication.q2_category_response[comment_id]

        if annotation is None:
            raise ValueError(f"Could not find annotation with ID {comment_id}")

        rows.append(
            CommentEntry(
                comment_id=annotation.id,
                annotator_id=annotation.annotator_id,
                accuracy_type=str(annotation.accuracy_type),
                certainty=str(annotation.element_correctness.certainty).lower(),
                risk=str(annotation.element_correctness.risk).lower(),
                urgency=str(annotation.element_correctness.urgency).lower(),
                categories=(
                    ", ".join(str(category) for category in categories) if categories else ""
                ),
                comment=annotation.comment,
            )
        )

    return rows


def calculate_answer_key_impact(section: SectionResponseAnnotation) -> str:
    """Calculate human-readable gloss re: impact of adjudication on answer key."""
    human_annotations = [
        annotation
        for annotation in section.annotations
        if annotation.annotator_id != "IsiHallucinationDetector"
    ]
    label_counter = Counter(annotation.accuracy_type for annotation in human_annotations)
    consensus_label, count = label_counter.most_common(n=1)[0]
    if section.adjudication is None:
        raise ValueError("No section adjudication to use to calculate impact.")
    elif count == label_counter.total() and consensus_label:
        impact_mapping = {
            ("correct", "no_hallucination"): "No Change",
            ("incorrect", "hallucination"): "No Change",
            (
                "disagreement",
                "hallucination",
            ): "Resolved Human Consensus Label of Disagreement: Hallucination",
            ("correct", "hallucination"): "New Hallucination",
            ("incorrect", "no_hallucination"): "Remove Hallucination",
            (
                "disagreement",
                "no_hallucination",
            ): "Resolved Human Consensus Label of Disagreement: No Hallucination",
        }
        result = impact_mapping[consensus_label, section.adjudication.aggregate_judgment]
    else:
        result = (
            "Resolved Human Disagreement: Hallucination"
            if section.adjudication.aggregate_judgment == "hallucination"
            else "Resolved Human Disagreement: No Hallucination"
        )
    return result


def build_html(responses: list[EnhancedResponse]) -> str:
    """Render list of responses with section-level adjudications as HTML."""
    response_entries: list[ResponseEntry] = []

    for response in responses:
        if response.annotations is None:
            continue

        section_entries: list[SectionEntry] = []
        for section in response.annotations.sections_with_annotations:
            adjudication = section.adjudication
            if adjudication is None:
                continue

            section_entries.append(
                SectionEntry(
                    section_id=section.id,
                    section_text=section.section,
                    aggregate_judgment=str(adjudication.aggregate_judgment),
                    q1_response=str(adjudication.q1_response),
                    q3_response=str(adjudication.q3_response),
                    adjudicator_id=adjudication.annotator_id,
                    adjudicator_comment=adjudication.comment,
                    answer_key_impact=calculate_answer_key_impact(section),
                    comment_rows=build_comment_rows(section),
                )
            )

        if section_entries:
            response_entries.append(
                ResponseEntry(
                    question=response.question,
                    response_id=response.id,
                    sections=section_entries,
                )
            )

    doc = document(title="Adjudication Summary")

    css = """
    /* Grouping box to clearly associate question, section, comments, and annotations. */
    .response-box {
      border: 2px solid #bbb;
      border-radius: 6px;
      padding: 10px 12px;
      margin: 18px 0 24px 0;
      background: #fcfcff;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    table.question-table, table.section-table, table.comments-table {
      border-collapse: collapse;
      border: 1px solid #666;
      width: 100%;
    }

    table.question-table th, table.question-table td,
    table.section-table th, table.section-table td,
    table.comments-table th, table.comments-table td {
      border: 1px solid #666;
      padding: 6px 8px;
      text-align: left;
      vertical-align: top;
      word-wrap: break-word;
      white-space: normal;
      max-width: 80vw;
    }

    /* Strong visual separation for questions */
    table.question-table {
      margin: 6px 0 10px 0;
      box-shadow: 0 1px 2px rgba(0,0,0,0.06);
      border: 2px solid #2c7be5;
    }

    table.question-table th.spanning,
    table.section-table th.spanning,
    table.comments-table th.spanning {
      text-align: left;
      background: #f0f0f0;
      font-weight: bold;
      padding: 8px;
    }

    table.question-table th.spanning {
      background: #e9f2ff;
      border-bottom: 2px solid #2c7be5;
      font-size: 1.05rem;
      font-weight: 700;
      padding: 10px;
    }

    table.section-table, table.comments-table {
      margin: 10px 0 20px 0;
    }
    """

    # Insert charset tag in <head> before <title> so title is interpreted properly
    # See: https://github.com/Knio/dominate/issues/140
    doc.children[0].children.insert(0, meta(charset="utf-8"))
    doc.add(style(css))

    # Total columns for the per-comment:
    # comment_id + annotator_id + 4 dimensions correctness + categories + comment text
    total_comment_cols = 1 + 1 + 4 + 1 + 1  # = 9

    for response_entry in response_entries:
        group = div(cls="response-box")

        header_tbl = table(cls="question-table")
        question_html = format_text("QUESTION:\n" + response_entry.question)
        top_tr = tr()
        top_th = th(colspan=2, _class="spanning")
        top_th.add_raw_string(question_html)
        top_tr.add(top_th)
        header_tbl.add(top_tr)
        header_tbl.add(tr(td("Response ID"), td(response_entry.response_id)))
        group.add(header_tbl)

        for section in response_entry.sections:
            section_tbl = table(cls="section-table")
            section_text_html = format_text("SECTION:\n" + section.section_text)
            top_tr = tr()
            top_th = th(colspan=2, _class="spanning")
            top_th.add_raw_string(section_text_html)
            top_tr.add(top_th)
            section_tbl.add(top_tr)
            section_tbl.add(tr(td("Section ID"), td(section.section_id)))
            group.add(section_tbl)

            comments_tbl = table(cls="comments-table")
            comments_tbl.add(
                tr(
                    th(
                        "Comments Shown To Adjudicator",
                        colspan=total_comment_cols,
                        _class="spanning",
                    )
                )
            )
            hdr_tr = tr()
            for header in [
                "comment_id",
                "annotator_id",
                "accuracy_type",
                "certainty",
                "risk",
                "urgency",
                "categories",
                "comment",
            ]:
                hdr_tr.add(th(header))
            comments_tbl.add(hdr_tr)

            for row in section.comment_rows:
                row_tr = tr()
                for value in [
                    row.comment_id,
                    row.annotator_id,
                    row.accuracy_type,
                    row.certainty,
                    row.risk,
                    row.urgency,
                    row.categories,
                    row.comment,
                ]:
                    cell = td()
                    cell.add_raw_string(format_text(value))
                    row_tr.add(cell)
                comments_tbl.add(row_tr)
            group.add(comments_tbl)

            adjudication_tbl = table(cls="section-table")
            adjudication_tbl.add(tr(th("Adjudication", colspan=2, _class="spanning")))
            for label, value in [
                ("Aggregate Judgment", section.aggregate_judgment),
                ("Q1", section.q1_response),
                ("Q3", section.q3_response),
                ("Adjudicator Comment", section.adjudicator_comment),
                ("Answer Key Impact", section.answer_key_impact),
            ]:
                value_cell = td()
                value_cell.add_raw_string(format_text(value))
                adjudication_tbl.add(tr(td(label), value_cell))
            group.add(adjudication_tbl)

        doc.body.add(group)

    return doc.render()


def main() -> None:  # noqa: D103
    parser = argparse.ArgumentParser(
        description="Show adjudicated section summaries using EnhancedResponse objects."
    )
    parser.add_argument("jsonl_file", type=Path, help="Path to JSONL with EnhancedResponse rows")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("adjudication_summary.html"),
        help="Output HTML path",
    )
    args = parser.parse_args()

    responses = [EnhancedResponse.model_validate(row) for row in read_jsonl(args.jsonl_file)]
    html = build_html(responses)
    args.output.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
