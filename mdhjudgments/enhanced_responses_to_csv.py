"""Convert EnhancedResponse JSONL data into normalized CSV tables."""

import argparse
from collections.abc import Iterable, Mapping, Sequence
import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, TypeVar

from mdhjudgments.file_utils import read_jsonl
from mdhjudgments.model import (
    AdjudicationCommentCategory,
    AnnotationModel,
    AnswerAnnotationModel,
    EnhancedResponse,
    IncorrectAnnotationType,
    InformationAccuracyAnnotationModel,
    SectionAdjudicationModel,
    SectionFactCheckingModel,
    SectionResponseAnnotation,
)
from mdhjudgments.run_hallucinations_analyses import is_human

CATEGORY_COLUMNS = [str(category) for category in AdjudicationCommentCategory]
DIMENSION_COLUMNS = ["Accuracy"]


ModelT = TypeVar("ModelT", bound=AnnotationModel)


@dataclass
class _MergedSection:
    """Data collected for one section across duplicate response records."""

    response_id: str
    section: str
    annotations: dict[str, InformationAccuracyAnnotationModel] = field(default_factory=dict)
    adjudications: dict[str, SectionAdjudicationModel] = field(default_factory=dict)
    fact_checkings: dict[str, SectionFactCheckingModel] = field(default_factory=dict)


@dataclass
class _MergedResponse:
    """Data collected for one response across duplicate input records."""

    response: EnhancedResponse
    answer_annotations: dict[str, AnswerAnnotationModel] = field(default_factory=dict)
    sections: dict[str, _MergedSection] = field(default_factory=dict)
    section_order: dict[str, int] = field(default_factory=dict)


def _add_unique(items: dict[str, ModelT], item: ModelT, *, label: str) -> None:
    """Add a model by ID, ignoring export-time timestamp differences."""
    previous = items.get(item.id)
    if previous is None:
        items[item.id] = item
    elif previous.model_dump(exclude={"timestamp"}) != item.model_dump(exclude={"timestamp"}):
        raise ValueError(f"Conflicting {label} data for ID {item.id!r}")


def _merge_responses(responses: Iterable[EnhancedResponse]) -> list[_MergedResponse]:
    """Merge annotation data belonging to duplicate responses and sections."""
    merged_responses: dict[str, _MergedResponse] = {}
    section_response_ids: dict[str, str] = {}

    for response in responses:
        merged_response = merged_responses.get(response.id)
        if merged_response is None:
            merged_response = _MergedResponse(response=response)
            merged_responses[response.id] = merged_response
        else:
            original = merged_response.response
            if original.question != response.question or original.answer != response.answer:
                raise ValueError(f"Conflicting response data for ID {response.id!r}")

        if response.annotations is None:
            continue

        for answer_annotation in response.annotations.answer_annotations:
            _add_unique(
                merged_response.answer_annotations,
                answer_annotation,
                label="answer annotation",
            )

        for idx, section in enumerate(response.annotations.sections_with_annotations):
            previous_response_id = section_response_ids.setdefault(section.id, response.id)
            if previous_response_id != response.id:
                raise ValueError(
                    f"Section ID {section.id!r} belongs to both response "
                    f"{previous_response_id!r} and {response.id!r}"
                )

            merged_section = merged_response.sections.get(section.id)
            if merged_section is None:
                merged_section = _MergedSection(
                    response_id=response.id,
                    section=section.section,
                )
                merged_response.sections[section.id] = merged_section
                merged_response.section_order[section.id] = idx
            else:
                if merged_section.section != section.section:
                    raise ValueError(f"Conflicting section text for ID {section.id!r}")
                if merged_response.section_order[section.id] != idx:
                    raise ValueError(f"Conflicting section ordering for ID {section.id!r}")

            for annotation in section.annotations:
                _add_unique(merged_section.annotations, annotation, label="first-pass annotation")
            if section.adjudication is not None:
                _add_unique(
                    merged_section.adjudications,
                    section.adjudication,
                    label="adjudication",
                )
            if section.fact_checking is not None:
                _add_unique(
                    merged_section.fact_checkings,
                    section.fact_checking,
                    label="fact-checking",
                )

    return list(merged_responses.values())


def _bool(*, value: bool) -> str:
    """Format a Boolean consistently for CSV output."""
    return str(value).lower()


def _has_hallucination(annotation_accuracy_type: IncorrectAnnotationType, **values: bool) -> str:
    """Calculate the requested first-pass hallucination indicator."""
    has_hallucination = (
        annotation_accuracy_type == IncorrectAnnotationType.FACTUALLY_INCORRECT
        or not values["certainty"]
        or not values["risk"]
        or not values["urgency"]
    )
    return _bool(value=has_hallucination)


def _has_claim(section: SectionResponseAnnotation) -> str:
    """Return the section's aggregate claim status when it can be determined."""
    claim_judgments = {
        annotation.accuracy_type != IncorrectAnnotationType.NO_FACT
        for annotation in section.annotations
        if annotation.accuracy_type != IncorrectAnnotationType.DISAGREEMENT
    }
    if len(claim_judgments) == 1:
        return _bool(value=claim_judgments.pop())
    return ""


def _write_csv(
    output_path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    """Write one CSV with a header, including when there are no data rows."""
    with open(output_path, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def convert(
    responses: Iterable[EnhancedResponse],
    output_dir: Path,
    *,
    include_first_pass_count: bool = False,
    include_has_claim: bool = False,
) -> None:
    """Convert EnhancedResponse objects into seven CSV tables.

    Args:
        responses: Responses to convert.
        output_dir: Directory in which to create the CSV files.
        include_first_pass_count: Add ``num_first_pass_annotations`` to sections.csv.
        include_has_claim: Add ``has_claim`` to sections.csv.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    question_response_rows: list[dict[str, Any]] = []
    section_rows: list[dict[str, Any]] = []
    response_judgment_rows: list[dict[str, Any]] = []
    first_pass_rows: list[dict[str, Any]] = []
    medical_section_rows: list[dict[str, Any]] = []
    medical_comment_rows: list[dict[str, Any]] = []
    factchecker_rows: list[dict[str, Any]] = []

    for merged_response in _merge_responses(responses):
        response = merged_response.response
        question_response_rows.append(
            {
                "response_id": response.id,
                "question": response.question,
                "response": response.answer,
            }
        )
        for answer_annotation in merged_response.answer_annotations.values():
            dimensions: dict[str, Any] = {}
            if answer_annotation.dimensions is not None:
                dimensions = answer_annotation.dimensions.model_dump(by_alias=True)
            response_judgment_rows.append(
                {
                    "response_id": response.id,
                    "annotator_id": answer_annotation.annotator_id,
                    **{column: dimensions.get(column) for column in DIMENSION_COLUMNS},
                }
            )

        for section_id, merged_section in merged_response.sections.items():
            section = SectionResponseAnnotation(
                id=section_id,
                section=merged_section.section,
                annotations=list(merged_section.annotations.values()),
            )
            section_row: dict[str, Any] = {
                "response_id": merged_section.response_id,
                "section_id": section.id,
                "section_order_in_response": merged_response.section_order[section.id],
                "section_text": section.section,
            }
            if include_first_pass_count:
                section_row["num_first_pass_annotations"] = len(
                    [annotation for annotation in section.annotations if is_human(annotation)]
                )
            if include_has_claim:
                section_row["has_claim"] = _has_claim(section)
            section_rows.append(section_row)

            for first_pass_annotation in section.annotations:
                correctness = first_pass_annotation.element_correctness
                first_pass_rows.append(
                    {
                        "section_id": section.id,
                        "annotation_id": first_pass_annotation.id,
                        "annotator_id": first_pass_annotation.annotator_id,
                        "is_human": is_human(first_pass_annotation),
                        "has_hallucination": _has_hallucination(
                            first_pass_annotation.accuracy_type,
                            certainty=correctness.certainty,
                            risk=correctness.risk,
                            urgency=correctness.urgency,
                        ),
                        "accuracy_type": str(first_pass_annotation.accuracy_type),
                        "certainty": _bool(value=correctness.certainty),
                        "risk": _bool(value=correctness.risk),
                        "urgency": _bool(value=correctness.urgency),
                    }
                )

            for adjudication in merged_section.adjudications.values():
                medical_section_rows.append(
                    {
                        "section_id": section.id,
                        "adjudicator_id": adjudication.annotator_id,
                        "aggregate_judgment": str(adjudication.aggregate_judgment),
                        "q1_response": str(adjudication.q1_response),
                        "q3_response": str(adjudication.q3_response),
                        "comment": adjudication.comment,
                        "annotation_ids_seen": json.dumps(adjudication.saw_annotation_ids),
                    }
                )
                for annotation_id, categories in adjudication.q2_category_response.items():
                    category_set = {str(category) for category in categories}
                    medical_comment_rows.append(
                        {
                            "annotation_id": annotation_id,
                            "medical_expert_adjudicator_id": adjudication.annotator_id,
                            **{
                                category: _bool(value=category in category_set)
                                for category in CATEGORY_COLUMNS
                            },
                        }
                    )

            for fact_checking in merged_section.fact_checkings.values():
                fact_categories = {
                    str(category.simple_category) for category in fact_checking.q2_category_response
                }
                other_text = [
                    category.other_text
                    for category in fact_checking.q2_category_response
                    if category.simple_category == AdjudicationCommentCategory.OTHER
                    and category.other_text is not None
                ]
                assert len(other_text) <= 1
                factchecker_rows.append(
                    {
                        "section_id": section.id,
                        "factchecker_id": fact_checking.annotator_id,
                        "aggregate_judgment": str(fact_checking.aggregate_judgment),
                        "q1_response": str(fact_checking.q1_response),
                        "q3_response": str(fact_checking.q3_choice),
                        "q3_url": fact_checking.q3_url,
                        "q4_excerpt": fact_checking.q4_excerpt,
                        "q5a_choice": str(fact_checking.q5a_choice),
                        "q5b_choice": str(fact_checking.q5b_choice),
                        "q5c_choice": (
                            str(fact_checking.q5c_choice.reason)
                            if fact_checking.q5c_choice is not None
                            else "n/a"
                        ),
                        "q6_comment": fact_checking.q6_comment,
                        **{
                            category: _bool(value=category in fact_categories)
                            for category in CATEGORY_COLUMNS
                        },
                        "Other text": "; ".join(other_text),
                        "annotation_ids_seen": json.dumps(fact_checking.saw_annotation_ids),
                    }
                )

    section_columns = ["response_id", "section_id", "section_order_in_response", "section_text"]
    if include_first_pass_count:
        section_columns.append("num_first_pass_annotations")
    if include_has_claim:
        section_columns.append("has_claim")

    _write_csv(
        output_dir / "questions_responses.csv",
        ["response_id", "question", "response"],
        question_response_rows,
    )
    _write_csv(output_dir / "sections.csv", section_columns, section_rows)
    _write_csv(
        output_dir / "response_level_annotations.csv",
        ["response_id", "annotator_id", *DIMENSION_COLUMNS],
        response_judgment_rows,
    )
    _write_csv(
        output_dir / "section_level_first_pass_annotations.csv",
        [
            "section_id",
            "annotation_id",
            "annotator_id",
            "is_human",
            "has_hallucination",
            "accuracy_type",
            "certainty",
            "risk",
            "urgency",
        ],
        first_pass_rows,
    )
    _write_csv(
        output_dir / "section_level_medical_expert_adjudications.csv",
        [
            "section_id",
            "adjudicator_id",
            "aggregate_judgment",
            "q1_response",
            "q3_response",
            "comment",
            "annotation_ids_seen",
        ],
        medical_section_rows,
    )
    _write_csv(
        output_dir / "comment_level_medical_expert_adjudication_judgments.csv",
        ["annotation_id", "medical_expert_adjudicator_id", *CATEGORY_COLUMNS],
        medical_comment_rows,
    )
    _write_csv(
        output_dir / "section_level_factchecker_adjudications.csv",
        [
            "section_id",
            "factchecker_id",
            "aggregate_judgment",
            "q1_response",
            "q3_response",
            "q3_url",
            "q4_excerpt",
            "q5a_choice",
            "q5b_choice",
            "q5c_choice",
            "q6_comment",
            *CATEGORY_COLUMNS,
            "Other text",
            "annotation_ids_seen",
        ],
        factchecker_rows,
    )


def main() -> None:
    """Run the command-line converter."""
    parser = argparse.ArgumentParser(
        description="Convert EnhancedResponse JSONL data into seven normalized CSV files."
    )
    parser.add_argument(
        "jsonl_files",
        nargs="+",
        type=Path,
        help="Input EnhancedResponse JSONL file(s)",
    )
    parser.add_argument("output_dir", type=Path, help="Directory for generated CSV files")
    parser.add_argument(
        "--include-first-pass-count",
        action="store_true",
        help="Add the number of first-pass annotations to each section row.",
    )
    parser.add_argument(
        "--include-has-claim",
        action="store_true",
        help="Add the aggregate has-claim judgment to each section row.",
    )
    args = parser.parse_args()

    responses = (
        EnhancedResponse.model_validate(row)
        for jsonl_file in args.jsonl_files
        for row in read_jsonl(jsonl_file)
    )
    convert(
        responses,
        args.output_dir,
        include_first_pass_count=args.include_first_pass_count,
        include_has_claim=args.include_has_claim,
    )


if __name__ == "__main__":
    main()
