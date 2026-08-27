"""Script to render fact-checking outcomes given fact-checked annotation data.

Takes a file of EnhancedResponses and renders all (and only) the sections that went through fact-checking.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
from html import escape
import json
import logging
from pathlib import Path

from dominate import document
from dominate.tags import div, meta, style, table, td, th, tr

from mdhjudgments.file_utils import read_jsonl
from mdhjudgments.model import (
    AdjudicationJudgment,
    EnhancedResponse,
    FactCheckingCommentsJudgment,
    FactCheckingContradictionReason,
    FactCheckingExcerptJudgment,
    FactCheckingUrlRelevance,
    IncorrectAnnotationType,
    InformationAccuracyAnnotationModel,
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
    comment: str


@dataclass
class SectionEntry:
    """Represents a section's annotation and fact-checking outcome, ready to be rendered as HTML."""

    section_id: str
    section_text: str
    q1_response: str
    q2_categories: str
    q3_choice: str
    q3_url: str
    q4_excerpt: str
    q5a_choice: str
    q5b_choice: str
    q5c_choice: str
    q6_comment: str
    fact_checker_id: str
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
    *,
    annotator_mapping: dict[str, str],
) -> list[CommentEntry]:
    """Build comment entries ready to be rendered as HTML."""
    assert section.fact_checking is not None

    annotations_by_id = {annotation.id: annotation for annotation in section.annotations}
    annotated_but_not_adjudicated = set(annotations_by_id.keys()) - set(
        section.fact_checking.saw_annotation_ids
    )
    if annotated_but_not_adjudicated:
        logger.debug(
            "Found %d comments annotated but not adjudicated --- IDs: %s",
            len(annotated_but_not_adjudicated),
            sorted(annotated_but_not_adjudicated),
        )

    rows: list[CommentEntry] = []

    for comment_id in section.fact_checking.saw_annotation_ids:
        annotation = annotations_by_id[comment_id]

        if annotation is None:
            raise ValueError(f"Could not find annotation with ID {comment_id}")

        anonymized_id = (
            annotator_mapping.get(
                annotation.annotator_id,
                f"UNMAPPED-{annotation.annotator_id[0]}{annotation.annotator_id[-1]}",
            )
            if annotation.annotator_id
            not in {"IsiHallucinationDetector", "IsiHallucinationInjector"}
            else annotation.annotator_id
        )
        rows.append(
            CommentEntry(
                comment_id=annotation.id,
                annotator_id=anonymized_id,
                accuracy_type=str(annotation.accuracy_type),
                certainty=str(annotation.element_correctness.certainty).lower(),
                risk=str(annotation.element_correctness.risk).lower(),
                urgency=str(annotation.element_correctness.urgency).lower(),
                comment=annotation.comment,
            )
        )

    return rows


def annotation_indicates_section_has_problem(
    annotation: InformationAccuracyAnnotationModel,
) -> bool:
    """Determine if a first-pass annotation indicates the section contains a hallucination."""
    return (
        annotation.accuracy_type == IncorrectAnnotationType.FACTUALLY_INCORRECT
        or not annotation.element_correctness.certainty
        or not annotation.element_correctness.risk
        or not annotation.element_correctness.urgency
    )


def calculate_answer_key_impact(section: SectionResponseAnnotation) -> str:
    """Calculate human-readable gloss re: impact of adjudication on answer key."""
    if section.fact_checking is None:
        raise ValueError("No section fact-checking to use to calculate impact.")

    factchecker_label: bool
    match (
        section.fact_checking.q5a_choice,
        section.fact_checking.q5b_choice,
        section.fact_checking.q5c_choice.reason if section.fact_checking.q5c_choice else None,
    ):
        # Evidence shows problem in excerpt
        case (
            (
                FactCheckingExcerptJudgment.EVIDENCE_SHOWS_INCORRECT_INFO_IN_EXCERPT
                | FactCheckingExcerptJudgment.LACK_OF_EVIDENCE_INDICATES_FABRICATION
            ),
            (
                FactCheckingCommentsJudgment.EVIDENCE_SUPPORTS_IDENTIFIED_ERROR
                | FactCheckingCommentsJudgment.EVIDENCE_SUPPORTS_IDENTIFIED_FABRICATION
                | FactCheckingCommentsJudgment.EVIDENCE_UNRELATED
                | FactCheckingCommentsJudgment.EVIDENCE_CONTRADICTS_COMMENTS
            ),
            _,
        ):
            factchecker_label = True
        # Evidence shows one of the comments correctly identifies a problem
        # and it's factual
        case (
            (
                FactCheckingExcerptJudgment.EVIDENCE_SHOWS_INCORRECT_INFO_IN_EXCERPT
                | FactCheckingExcerptJudgment.LACK_OF_EVIDENCE_INDICATES_FABRICATION
                | FactCheckingExcerptJudgment.EVIDENCE_FULLY_SUPPORTS_EXCERPT
                | FactCheckingExcerptJudgment.EVIDENCE_UNRELATED
            ),
            (
                FactCheckingCommentsJudgment.EVIDENCE_SUPPORTS_IDENTIFIED_ERROR
                | FactCheckingCommentsJudgment.EVIDENCE_SUPPORTS_IDENTIFIED_FABRICATION
            ),
            (
                None
                | FactCheckingContradictionReason.JUDGE_INACCURACIES_TOO_MINOR_TO_MATTER
                | FactCheckingContradictionReason.OTHER
            ),
        ):
            factchecker_label = True
        # Evidence shows no problem
        case (
            (
                FactCheckingExcerptJudgment.EVIDENCE_FULLY_SUPPORTS_EXCERPT
                | FactCheckingExcerptJudgment.EVIDENCE_UNRELATED
            ),
            (
                FactCheckingCommentsJudgment.EVIDENCE_CONTRADICTS_COMMENTS
                | FactCheckingCommentsJudgment.EVIDENCE_UNRELATED
            ),
            None,
        ):
            factchecker_label = False
        # Evidence shows no factual problem
        case (
            (
                FactCheckingExcerptJudgment.EVIDENCE_FULLY_SUPPORTS_EXCERPT
                | FactCheckingExcerptJudgment.EVIDENCE_UNRELATED
            ),
            _,
            FactCheckingContradictionReason.COMMENTS_NOT_ABOUT_FACTUAL_ACCURACY,
        ):
            factchecker_label = False
        case _, _, _:
            raise ValueError(
                f"Unhandled case: {section.fact_checking.q5a_choice}, {section.fact_checking.q5b_choice}, {section.fact_checking.q5c_choice}"
            )

    human_annotations = [
        annotation
        for annotation in section.annotations
        if annotation.annotator_id not in {"IsiHallucinationDetector", "IsiHallucinationInjector"}
    ]
    label_counter = Counter(
        annotation_indicates_section_has_problem(annotation) for annotation in human_annotations
    )
    consensus_label, count = label_counter.most_common(n=1)[0]
    if count == label_counter.total() and consensus_label:
        # We have human annotator consensus
        impact_mapping = {
            (False, True): "New Hallucination",
            (True, False): "Remove Hallucination",
            (False, False): "No Change",
            (True, True): "No Change",
        }
        return impact_mapping[consensus_label, factchecker_label]
    else:
        # We have human annotator disagreement
        result = (
            "Resolved Human Disagreement: No Hallucination"
            if not factchecker_label
            else "Resolved Human Disagreement: Hallucination"
        )
    return result


def build_html(responses: list[EnhancedResponse], *, annotator_mapping: dict[str, str]) -> str:
    """Render list of responses with section-level adjudications as HTML."""
    response_entries: list[ResponseEntry] = []

    for response in responses:
        if response.annotations is None:
            continue

        section_entries: list[SectionEntry] = []
        for section in response.annotations.sections_with_annotations:
            fact_checking = section.fact_checking
            if fact_checking is None:
                continue

            q5c_choice: str = "n/a"
            if fact_checking.q5c_choice:
                q5c_choice = str(fact_checking.q5c_choice.reason)
                if fact_checking.q5c_choice.explanation:
                    q5c_choice += f" ({fact_checking.q5c_choice.explanation})"

            section_entries.append(
                SectionEntry(
                    section_id=section.id,
                    section_text=section.section,
                    q1_response=str(fact_checking.q1_response),
                    q2_categories=", ".join(
                        str(category.simple_category)
                        + (f" ({category.other_text})" if category.other_text else "")
                        for category in fact_checking.q2_category_response
                    ),
                    q3_choice=str(fact_checking.q3_choice),
                    q3_url=str(fact_checking.q3_url),
                    q4_excerpt=str(fact_checking.q4_excerpt),
                    q5a_choice=str(fact_checking.q5a_choice),
                    q5b_choice=str(fact_checking.q5b_choice),
                    q5c_choice=q5c_choice,
                    q6_comment=str(fact_checking.q6_comment),
                    fact_checker_id=fact_checking.annotator_id,
                    answer_key_impact=calculate_answer_key_impact(section),
                    comment_rows=build_comment_rows(section, annotator_mapping=annotator_mapping),
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

    doc = document(title="Fact-Checking Summary")

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

    td.label {
      width: 10em;
    }

    td.gloss {
      width: 30em;
    }
    """

    # Insert charset tag in <head> before <title> so title is interpreted properly
    # See: https://github.com/Knio/dominate/issues/140
    doc.children[0].children.insert(0, meta(charset="utf-8"))
    doc.add(style(css))

    # Total columns for the per-comment:
    # comment_id + annotator_id + 4 dimensions correctness + comment text
    total_comment_cols = 1 + 1 + 4 + 1  # = 8

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
                        "Comments Shown To Fact-Checker",
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
                    row.comment,
                ]:
                    cell = td()
                    cell.add_raw_string(format_text(value))
                    row_tr.add(cell)
                comments_tbl.add(row_tr)
            group.add(comments_tbl)

            fact_checking_tbl = table(cls="section-table")
            fact_checking_tbl.add(tr(th("Fact-Checking", colspan=3, _class="spanning")))
            fact_checking_tbl.add(
                tr(th("Value ID", th("Rough summary of question/instructions"), th("Value")))
            )
            q5a_value_to_q6_gloss = {
                str(
                    FactCheckingExcerptJudgment.EVIDENCE_SHOWS_INCORRECT_INFO_IN_EXCERPT
                ): "Describe how evidence refutes information in chatbot excerpt and how it relates to the comments",
                str(
                    FactCheckingExcerptJudgment.EVIDENCE_FULLY_SUPPORTS_EXCERPT
                ): "Describe how evidence supports information in chatbot excerpt and how it relates to the comments",
                str(
                    FactCheckingExcerptJudgment.EVIDENCE_UNRELATED
                ): "Describe the kind of evidence you would have needed to support or refute the claim, and describe why it was hard to find",
                str(
                    FactCheckingExcerptJudgment.LACK_OF_EVIDENCE_INDICATES_FABRICATION
                ): "Describe search process you used to determine there was a fabrication",
            }
            q6_gloss = q5a_value_to_q6_gloss[section.q5a_choice]
            for label, gloss, value in [
                (
                    "Q1",
                    "Which best describes your opinion about the accuracy of the chatbot excerpt?",
                    section.q1_response,
                ),
                (
                    "Q2 Categories",
                    "Categorize the displayed comments as a whole&mdash;you may select multiple categories <br><br>"
                    "[AT LEAST ONE OF: (A) Citations, (B) Numeric Info, (C) Other Medical Info, (D) Missing Info, "
                    "(E) Clarity, (F) Other. Please describe. (text box provided)]",
                    section.q2_categories,
                ),
                (
                    "Q3 Choice",
                    "Look for evidence that supports or refutes the information in the excerpt and describe "
                    "its relationship to the excerpt&mdash;if you can't find evidence that supports or refutes, "
                    "look for a URL that answers some aspect of the original question<br><br>"
                    "[ONE OF: (A) supports or refutes, (B) hallucination is a fabrication, OR "
                    "(C) could not find evidence for other reasons, but this was broadly relevant]",
                    section.q3_choice,
                ),
                ("Q3 URL", "Provide the URL for the evidence you found", section.q3_url),
                (
                    "Q4 Evidence Sentences",
                    "Provide 1-6 relevant sentences from your evidence URL that are relevant to judging correctness of the chatbot excerpt, or otherwise relevant to the question",
                    section.q4_excerpt,
                ),
                (
                    "Q5a Choice",
                    "Describe relationship of evidence to excerpt's claims<br><br>"
                    "[ONE OF: (A) evidence shows excerpt contains incorrect information, (B) excerpt contains fabricated claims, (C) evidence fully supports excerpt, OR (D) could not find evidence that verifies or refutes excerpt's claims.]",
                    section.q5a_choice,
                ),
                (
                    "Q5b Choice",
                    "Describe relationship of evidence to comments<br><br>"
                    "[ONE OF: (A) evidence supports comment-identified errors, (B) evidence supports comment-identified fabrications, (C) evidence contradicts comments, OR (D) evidence unrelated to comments.]",
                    section.q5b_choice,
                ),
                (
                    "Q5c Choice",
                    "(Only asked if Q5a answer was fully supports and Q5b said the comments correctly identify an error or fabrication.) Explain contradiction between Q5a and Q5b<br><br>"
                    "[ONE OF: (A) comments are not about factual accuracy, (B) the inaccuracies are too minor to matter, OR (C) other, specified in text.]",
                    section.q5c_choice,
                ),
                ("Q6 Comment", q6_gloss, section.q6_comment),
                (
                    "Answer Key Impact",
                    "*Summary of fact-checker opinion as described in above questions (*NOT a separate question posed to fact-checker)",
                    section.answer_key_impact,
                ),
            ]:
                value_cell = td()
                value_remap: dict[str | AdjudicationJudgment | FactCheckingUrlRelevance, str] = {
                    AdjudicationJudgment.MAJOR: "Major hallucination",
                    AdjudicationJudgment.MINOR: "Minor hallucination",
                    AdjudicationJudgment.NO_HALLUCINATION: "No hallucination",
                    AdjudicationJudgment.NEEDS_RESEARCH: "I would need to do research to judge the excerpt's accuracy",
                    FactCheckingUrlRelevance.SUPPORTS_OR_REFUTES: "URL supports or refutes information in chatbot excerpt",
                    FactCheckingUrlRelevance.UNSUPPORTED_FABRICATION: "Chatbot excerpt contains fabricated inormation I couldn't verify, but this URL is relevant"
                    "to the question.",
                    FactCheckingUrlRelevance.BROADLY_RELEVANT: "I could not find a reference for other reasons, but this URL was broadly relevant to the question.",
                    FactCheckingExcerptJudgment.EVIDENCE_SHOWS_INCORRECT_INFO_IN_EXCERPT: "The evidence indicates that the excerpt contains incorrect information.",
                    FactCheckingExcerptJudgment.LACK_OF_EVIDENCE_INDICATES_FABRICATION: "The lack of evidence indicates that the excerpt contains a fabrication.",
                    FactCheckingExcerptJudgment.EVIDENCE_FULLY_SUPPORTS_EXCERPT: "The evidence indicates that the excerpt is fully correct.",
                    FactCheckingExcerptJudgment.EVIDENCE_UNRELATED: "I could not find evidence that verifies or refutes the excerpt's claims.",
                    FactCheckingCommentsJudgment.EVIDENCE_SUPPORTS_IDENTIFIED_ERROR: "One or more comments correctly indicates an error according to the evidence.",
                    FactCheckingCommentsJudgment.EVIDENCE_SUPPORTS_IDENTIFIED_FABRICATION: "One or more comments identify a fabrication, and the lack of evidence for the problematic claim supports that.",
                    FactCheckingCommentsJudgment.EVIDENCE_CONTRADICTS_COMMENTS: "The comments are contradicted by the evidence, so the excerpt is correct.",
                    FactCheckingCommentsJudgment.EVIDENCE_UNRELATED: "The comments are unrelated to the evidence.",
                    FactCheckingContradictionReason.COMMENTS_NOT_ABOUT_FACTUAL_ACCURACY: "The comment(s) are not about factual accuracy.",
                    FactCheckingContradictionReason.JUDGE_INACCURACIES_TOO_MINOR_TO_MATTER: "The comment(s) indicate inaccuracies, but I consider the inaccuracies too minor to matter.",
                    FactCheckingContradictionReason.OTHER: "Other, please specify",
                }
                value_cell.add_raw_string(
                    f"{format_text(value)} ({format_text(value_remap[value])})"
                    if value in value_remap
                    else format_text(value)
                )
                gloss_cell = td(_class="gloss")
                gloss_cell.add_raw_string(gloss)
                fact_checking_tbl.add(tr(td(label, _class="label"), gloss_cell, value_cell))
            group.add(fact_checking_tbl)

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
        default=Path("fact_checking_summary.html"),
        help="Output HTML path",
    )
    parser.add_argument(
        "--mapping-path",
        type=Path,
        help="Where to load the mapping path used to anonymize annotator IDs.",
    )
    args = parser.parse_args()

    if args.mapping_path:
        with open(args.mapping_path, encoding="utf-8") as mapping_file:
            annotator_mapping = json.load(mapping_file)
    else:
        annotator_mapping = {}

    responses = [EnhancedResponse.model_validate(row) for row in read_jsonl(args.jsonl_file)]
    html = build_html(responses, annotator_mapping=annotator_mapping)
    args.output.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
