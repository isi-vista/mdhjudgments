"""Script to run EMNLP 2026 hallucinations paper analyses."""

import argparse
from collections import Counter
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field
from itertools import pairwise
import math
from pathlib import Path
from shutil import copy2
from typing import Any, Literal, TypeVar, cast

import krippendorff
from matplotlib.colors import BoundaryNorm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from mdhjudgments.model import (
    AdjudicationCommentCategory,
    AdjudicationJudgment,
    AnnotationModel,
    EnhancedResponse,
    FactCheckingCommentsJudgment,
    FactCheckingExcerptJudgment,
    IncorrectAnnotationType,
    InformationAccuracyAnnotationModel,
    SectionAdjudicationModel,
    SectionFactCheckingModel,
    SectionResponseAnnotation,
)
from mdhjudgments.xrr import DistanceFunctions, XrrMetrics, compute_xrr_with_summary_datasets


def is_human(annotation: AnnotationModel) -> bool:
    """Check if annotation was done by a human."""
    return (
        annotation.annotator_id not in (LAJ_ID, INJECTOR_ID)
        and "detector" not in annotation.annotator_id.lower()
        and "injector" not in annotation.annotator_id.lower()
    )


def calculate_krippendorff(
    annotators2items: dict[str, dict[str, int | None]], *, level: str
) -> float:
    """Calculate Krippendorff's alpha from a mapping.

    Mapping goes:
    annotator ID -> {item ID -> label}
    """
    reliability_data = [
        [int(value) if value is not None else np.nan for value in items2values.values()]
        for items2values in annotators2items.values()
    ]
    return krippendorff.alpha(reliability_data=reliability_data, level_of_measurement=level)


def calculate_response_accuracy_krippendorff(responses: list[EnhancedResponse]) -> float:
    """Calculate Krippendorff's alpha over response-level accuracy annotations."""
    annotators = set()
    for response in responses:
        for annotation in response.annotations.answer_annotations:
            assert is_human(annotation)
            annotators.add(annotation.annotator_id)

    annotators2responses = {}
    for response in responses:
        for annotation in response.annotations.answer_annotations:
            assert is_human(annotation)
            responses2values = annotators2responses.setdefault(annotation.annotator_id, {})
            assert response.id not in responses2values
            responses2values[response.id] = int(annotation.dimensions.accuracy)
        for annotator in annotators:
            annotators2responses.setdefault(annotator, {}).setdefault(response.id, None)

    return calculate_krippendorff(annotators2responses, level="ordinal")


def calculate_no_claim_krippendorff(sections: list[SectionResponseAnnotation]) -> float:
    """Calculate Krippendorff's alpha on annotators' claim vs. not-claim judgments."""
    annotators = set()
    for section in sections:
        for annotation in section.annotations:
            if is_human(annotation):
                annotators.add(annotation.annotator_id)

    annotators2sections = {}
    for section in sections:
        for annotation in section.annotations:
            if is_human(annotation):
                sections2values = annotators2sections.setdefault(annotation.annotator_id, {})
                assert section.id not in sections2values
                sections2values[section.id] = int(
                    annotation.accuracy_type == IncorrectAnnotationType.NO_FACT
                )
        for annotator in sorted(annotators):
            annotators2sections.setdefault(annotator, {}).setdefault(section.id, None)

    return calculate_krippendorff(annotators2sections, level="nominal")


def says_has_error(annotation: InformationAccuracyAnnotationModel) -> bool:
    """Return true if the annotation marks the section as having an error."""
    return (
        annotation.accuracy_type == IncorrectAnnotationType.FACTUALLY_INCORRECT
        or not annotation.element_correctness.certainty
        or not annotation.element_correctness.risk
        or not annotation.element_correctness.urgency
    )


def calculate_has_error_krippendorff(
    sections: list[SectionResponseAnnotation], *, check_is_human: bool
) -> float:
    """Calculate Krippendorff's alpha on annotators' claim vs. not-claim judgments."""
    annotators = set()
    for section in sections:
        for annotation in section.annotations:
            if is_human(annotation) or not check_is_human:
                annotators.add(annotation.annotator_id)

    annotators2sections = {}
    for section in sections:
        for annotation in section.annotations:
            if (is_human(annotation) or not check_is_human) and not any(
                is_human(annotation) and annotation.accuracy_type == IncorrectAnnotationType.NO_FACT
                for annotation in section.annotations
            ):
                sections2values = annotators2sections.setdefault(annotation.annotator_id, {})
                assert section.id not in sections2values
                sections2values[section.id] = int(says_has_error(annotation))
        for annotator in sorted(annotators):
            annotators2sections.setdefault(annotator, {}).setdefault(section.id, None)

    return calculate_krippendorff(annotators2sections, level="nominal")


def calculate_mean_accuracy(annotation_data: list[EnhancedResponse]) -> float:
    """Calculate mean accuracy."""
    return float(
        np.mean(
            [
                annotation.dimensions.accuracy
                for response in annotation_data
                for annotation in response.annotations.answer_annotations
                if is_human(annotation)
                # Can't assert in a list comprehension but this should be true
                # assert annotation.dimensions.accuracy is not None
            ]
        )
    )


def response_level_statistics(
    annotation_data: list[EnhancedResponse],
) -> dict[str, int | float | None]:
    """Calculate response-level statistics."""
    n_annotators_per_response: list[int] = [
        len(
            [
                annotation
                for annotation in response.annotations.answer_annotations
                if is_human(annotation)
            ]
        )
        for response in annotation_data
    ]
    n_annotators_per_section: list[int] = [
        len([annotation for annotation in section.annotations if is_human(annotation)])
        for response in annotation_data
        for section in response.annotations.sections_with_annotations
    ]
    n_sections_per_response = [
        len(response.annotations.sections_with_annotations) for response in annotation_data
    ]
    n_with_no_annotations = len(
        [
            response
            for response in annotation_data
            if len(response.annotations.answer_annotations) == 0
        ]
    )
    n_with_single_annotator = len(
        [
            response
            for response in annotation_data
            if len(response.annotations.answer_annotations) == 1
        ]
    )
    n_with_multiple_annotators = len(
        [
            response
            for response in annotation_data
            if len(response.annotations.answer_annotations) >= 2
        ]
    )
    n_with_no_human_section_annotations = len(
        [
            response
            for response in annotation_data
            if len(
                [
                    a
                    for a in response.annotations.sections_with_annotations[0].annotations
                    if is_human(a)
                ]
            )
            == 0
        ]
    )
    n_with_single_annotator_per_section = len(
        [
            response
            for response in annotation_data
            if len(
                [
                    a
                    for a in response.annotations.sections_with_annotations[0].annotations
                    if is_human(a)
                ]
            )
            == 1
        ]
    )
    n_with_multiple_annotators_per_section = len(
        [
            response
            for response in annotation_data
            if len(
                [
                    a
                    for a in response.annotations.sections_with_annotations[0].annotations
                    if is_human(a)
                ]
            )
            >= 2
        ]
    )
    return {
        "n_responses": len(annotation_data),
        "n_responses_with_no_annotations": n_with_no_annotations,
        "n_responses_with_no_section_annotations": n_with_no_human_section_annotations,
        "n_responses_single_annotated": n_with_single_annotator,
        "n_responses_single_annotated_per_section": n_with_single_annotator_per_section,
        "n_responses_multiple_annotated": n_with_multiple_annotators,
        "n_responses_multiple_annotated_per_section": n_with_multiple_annotators_per_section,
        "min_response_annotators": min(n_annotators_per_response),
        "max_response_annotators": max(n_annotators_per_response),
        "mean_response_annotators": float(np.mean(n_annotators_per_response)),
        "min_section_annotators": min(n_annotators_per_section),
        "max_section_annotators": max(n_annotators_per_section),
        "mean_section_annotators": float(np.mean(n_annotators_per_section)),
        "min_sections": min(n_sections_per_response),
        "max_sections": max(n_sections_per_response),
        "q25_sections": float(np.percentile(n_sections_per_response, 25)),
        "median_sections": float(np.median(n_sections_per_response)),
        "q75_sections": float(np.percentile(n_sections_per_response, 75)),
        "mean_sections": float(np.mean(n_sections_per_response)),
        "mean_accuracy_value": calculate_mean_accuracy(annotation_data),
        "accuracy_agreement": calculate_response_accuracy_krippendorff(annotation_data),
    }


LAJ_ID = "IsiHallucinationDetector"
INJECTOR_ID = "IsiHallucinationInjector"

FIRST_PASS_ANNOTATOR_TYPES = {
    "ANNOTATOR-0": "AI Researcher",
    "ANNOTATOR-1": "AI Researcher",
    "ANNOTATOR-14": "Student",
    "EXPERT-2": "Medical Expert",
    "EXPERT-3": "Medical Expert",
    "EXPERT-15": "Medical Expert",
    "EXPERT-4": "Medical Expert",
    "EXPERT-5": "Medical Expert",
    "ANNOTATOR-6": "Student",
    "ANNOTATOR-7": "Student",
    "ANNOTATOR-9": "AI Researcher",
    "ANNOTATOR-10": "AI Researcher",
    "ANNOTATOR-17": "Student",
    "EXPERT-18": "Medical Expert",
    "EXPERT-19": "Medical Expert",
    "ANNOTATOR-11": "AI Researcher",
    "ANNOTATOR-12": "Student",
    "ANNOTATOR-20": "Student",
    "ANNOTATOR-13": "AI Researcher",
    "EXPERT-21": "Medical Expert",
}

ANNOTATOR_TYPE_ORDER = ("AI Researcher", "Student", "Medical Expert")
COMMENT_REVIEWER_TYPE_ORDER = ("AI Researcher", "Student", "Medical Expert", "LaJ")
FIRST_PASS_SCORE_ANSWER_KEYS = (
    "first_pass_annotators",
    "first_pass_annotators_plus_medical_experts",
    "first_pass_annotators_plus_fact_checkers",
)
MIN_FIRST_PASS_ANNOTATIONS_FOR_SCORES = 2
DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_BOOTSTRAP_SEED = 42
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95

# Models that generated the annotated responses, in display order. The generating model appears as
# a substring of the response ID (e.g. "...-Qwen/Qwen3-32b-").
RESPONSE_MODELS = ("Qwen/Qwen3-32b", "google/gemma-3-12b-it", "gpt-4.1")

# Augmentation used when generating a response, keyed by the suffix its ID ends with. Ordered so
# that the more specific suffixes are checked before the bare "-" (which means no augmentation).
RESPONSE_AUGMENTATION_SUFFIXES = (
    ("-_-_be_brief", "be brief"),
    ("-_-_provide_medical_evidence", "provide medical evidence"),
    ("-version='hallucination_2025-06-07'", "hallucination injection"),
    ("-", "none"),
)
RESPONSE_AUGMENTATION_ORDER = (
    "none",
    "be brief",
    "provide medical evidence",
    "hallucination injection",
)


def parse_response_model_and_augmentation(response_id: str) -> tuple[str, str]:
    """Parse the generating model and augmentation out of a response ID."""
    model = next((model for model in RESPONSE_MODELS if f"-{model}-" in response_id), None)
    if model is None:
        raise ValueError(f"Could not determine generating model from response ID {response_id!r}.")

    augmentation = next(
        (name for suffix, name in RESPONSE_AUGMENTATION_SUFFIXES if response_id.endswith(suffix)),
        None,
    )
    if augmentation is None:
        raise ValueError(f"Could not determine augmentation from response ID {response_id!r}.")

    return model, augmentation


def response_counts_by_model_and_augmentation(
    annotation_data: list[EnhancedResponse],
) -> list[dict[str, str | int]]:
    """Break down responses by generating model and augmentation, with row and column totals."""
    counts: dict[str, dict[str, int]] = {
        model: dict.fromkeys(RESPONSE_AUGMENTATION_ORDER, 0) for model in RESPONSE_MODELS
    }
    for response in annotation_data:
        model, augmentation = parse_response_model_and_augmentation(response.id)
        counts[model][augmentation] += 1

    rows: list[dict[str, str | int]] = []
    for model in RESPONSE_MODELS:
        row: dict[str, str | int] = {"model": model, **counts[model]}
        row["total"] = sum(counts[model].values())
        rows.append(row)

    total_row: dict[str, str | int] = {"model": "total"}
    for augmentation in RESPONSE_AUGMENTATION_ORDER:
        total_row[augmentation] = sum(counts[model][augmentation] for model in RESPONSE_MODELS)
    total_row["total"] = sum(
        total_row[augmentation] for augmentation in RESPONSE_AUGMENTATION_ORDER
    )
    rows.append(total_row)

    return rows


@dataclass
class _ScoreAccumulator:
    """Collect binary labels for one first-pass annotator type."""

    golds: list[bool] = field(default_factory=list)
    preds: list[bool] = field(default_factory=list)
    section_ids: list[str] = field(default_factory=list)
    response_ids: list[str] = field(default_factory=list)


def first_pass_annotator_type(
    annotation: AnnotationModel | InformationAccuracyAnnotationModel,
) -> str | None:
    """Return the configured first-pass annotator type for an annotation."""
    return FIRST_PASS_ANNOTATOR_TYPES.get(annotation.annotator_id)


def _first_pass_section_annotations(
    section: SectionResponseAnnotation,
) -> list[InformationAccuracyAnnotationModel]:
    """Return first-pass annotations for a section."""
    return [
        annotation
        for annotation in section.annotations
        if first_pass_annotator_type(annotation) is not None
    ]


def count_distinct_first_pass_annotators_by_type(
    annotation_data: list[EnhancedResponse],
) -> tuple[int, dict[str, int]]:
    """Count distinct first-pass annotators overall and by annotator type."""
    annotator_ids_by_type: dict[str, set[str]] = {
        annotator_type: set() for annotator_type in ANNOTATOR_TYPE_ORDER
    }
    for response in annotation_data:
        if response.annotations is None:
            continue

        annotations = list(response.annotations.answer_annotations)
        for section in response.annotations.sections_with_annotations:
            annotations.extend(section.annotations)

        for annotation in annotations:
            annotator_type = first_pass_annotator_type(annotation)
            if annotator_type is not None:
                annotator_ids_by_type[annotator_type].add(annotation.annotator_id)

    return sum(len(ids) for ids in annotator_ids_by_type.values()), {
        annotator_type: len(annotator_ids_by_type[annotator_type])
        for annotator_type in ANNOTATOR_TYPE_ORDER
    }


def distinct_first_pass_annotators_by_type_and_level(
    annotation_data: list[EnhancedResponse],
) -> list[dict[str, str | int]]:
    """Count distinct first-pass annotators by annotator type and annotation level."""
    response_level_annotator_ids_by_type: dict[str, set[str]] = {
        annotator_type: set() for annotator_type in ANNOTATOR_TYPE_ORDER
    }
    section_level_annotator_ids_by_type: dict[str, set[str]] = {
        annotator_type: set() for annotator_type in ANNOTATOR_TYPE_ORDER
    }

    for response in annotation_data:
        if response.annotations is None:
            continue

        for annotation in response.annotations.answer_annotations:
            annotator_type = first_pass_annotator_type(annotation)
            if annotator_type is not None:
                response_level_annotator_ids_by_type[annotator_type].add(annotation.annotator_id)

        for section in response.annotations.sections_with_annotations:
            for annotation in section.annotations:
                annotator_type = first_pass_annotator_type(annotation)
                if annotator_type is not None:
                    section_level_annotator_ids_by_type[annotator_type].add(annotation.annotator_id)

    return [
        {
            "annotator_type": annotator_type,
            "response_level_annotators": len(response_level_annotator_ids_by_type[annotator_type]),
            "section_level_annotators": len(section_level_annotator_ids_by_type[annotator_type]),
            "any_level_annotators": len(
                response_level_annotator_ids_by_type[annotator_type]
                | section_level_annotator_ids_by_type[annotator_type]
            ),
        }
        for annotator_type in ANNOTATOR_TYPE_ORDER
    ]


def first_pass_annotation_coverage_by_type(
    annotation_data: list[EnhancedResponse],
) -> list[dict[str, str | int]]:
    """Count first-pass coverage by annotator type, level, and annotation multiplicity."""
    annotator_ids_by_type: dict[str, set[str]] = {
        annotator_type: set() for annotator_type in ANNOTATOR_TYPE_ORDER
    }
    single_annotated_response_ids_by_type: dict[str, set[str]] = {
        annotator_type: set() for annotator_type in ANNOTATOR_TYPE_ORDER
    }
    multiple_annotated_response_ids_by_type: dict[str, set[str]] = {
        annotator_type: set() for annotator_type in ANNOTATOR_TYPE_ORDER
    }
    section_level_single_annotated_response_ids_by_type: dict[str, set[str]] = {
        annotator_type: set() for annotator_type in ANNOTATOR_TYPE_ORDER
    }
    section_level_multiple_annotated_response_ids_by_type: dict[str, set[str]] = {
        annotator_type: set() for annotator_type in ANNOTATOR_TYPE_ORDER
    }
    single_annotated_section_ids_by_type: dict[str, set[str]] = {
        annotator_type: set() for annotator_type in ANNOTATOR_TYPE_ORDER
    }
    multiple_annotated_section_ids_by_type: dict[str, set[str]] = {
        annotator_type: set() for annotator_type in ANNOTATOR_TYPE_ORDER
    }

    n_response_judgments = Counter()
    n_section_judgments = Counter()
    for response in annotation_data:
        if response.annotations is None:
            continue

        response_level_types: set[str] = set()
        response_level_annotations = []
        for annotation in response.annotations.answer_annotations:
            annotator_type = first_pass_annotator_type(annotation)
            if annotator_type is not None:
                n_response_judgments[annotator_type] += 1
                annotator_ids_by_type[annotator_type].add(annotation.annotator_id)
                response_level_types.add(annotator_type)
                response_level_annotations.append(annotation)

        response_ids_by_type = (
            single_annotated_response_ids_by_type
            if len(response_level_annotations) == 1
            else multiple_annotated_response_ids_by_type
        )
        if response_level_annotations:
            for annotator_type in response_level_types:
                response_ids_by_type[annotator_type].add(response.id)

        get_section_level_response_numbers = False
        for section in response.annotations.sections_with_annotations:
            if any(
                annotation.accuracy_type == IncorrectAnnotationType.NO_FACT
                for annotation in section.annotations
            ):
                continue

            section_level_types: set[str] = set()
            section_level_annotations = []
            for annotation in section.annotations:
                annotator_type = first_pass_annotator_type(annotation)
                if annotator_type is not None:
                    n_section_judgments[annotator_type] += 1
                    annotator_ids_by_type[annotator_type].add(annotation.annotator_id)
                    section_level_types.add(annotator_type)
                    section_level_annotations.append(annotation)

            section_ids_by_type = (
                single_annotated_section_ids_by_type
                if len(section_level_annotations) == 1
                else multiple_annotated_section_ids_by_type
            )
            section_level_response_ids_by_type = (
                section_level_single_annotated_response_ids_by_type
                if len(section_level_annotations) == 1
                else section_level_multiple_annotated_response_ids_by_type
            )
            if section_level_annotations:
                for annotator_type in section_level_types:
                    section_ids_by_type[annotator_type].add(section.id)
                    if get_section_level_response_numbers:
                        section_level_response_ids_by_type[annotator_type].add(response.id)
            get_section_level_response_numbers = False

    return [
        {
            "annotator_type": annotator_type,
            "distinct_annotators": len(annotator_ids_by_type[annotator_type]),
            "n_response_judgments": n_response_judgments[annotator_type],
            "n_section_judgments": n_section_judgments[annotator_type],
            "response_level_single_annotated_responses": len(
                single_annotated_response_ids_by_type[annotator_type]
            ),
            "section_level_single_annotated_responses": len(
                section_level_single_annotated_response_ids_by_type[annotator_type]
            ),
            "response_level_multiple_annotated_responses": len(
                multiple_annotated_response_ids_by_type[annotator_type]
            ),
            "section_level_multiple_annotated_responses": len(
                section_level_multiple_annotated_response_ids_by_type[annotator_type]
            ),
            "section_level_single_annotated_sections": len(
                single_annotated_section_ids_by_type[annotator_type]
            ),
            "section_level_multiple_annotated_sections": len(
                multiple_annotated_section_ids_by_type[annotator_type]
            ),
        }
        for annotator_type in ANNOTATOR_TYPE_ORDER
    ]


def _consensus_bool(labels: Iterable[bool]) -> bool | None:
    """Return the unanimous boolean label, or None when labels are empty or disagree."""
    label_list = list(labels)
    if not label_list or len(set(label_list)) != 1:
        return None
    return label_list[0]


def _first_pass_section_scoreability(
    section: SectionResponseAnnotation,
) -> tuple[bool, bool | None, list[InformationAccuracyAnnotationModel]]:
    """Return scoreability, first-pass consensus, and first-pass annotations for a section."""
    first_pass_annotations = _first_pass_section_annotations(section)
    if len(first_pass_annotations) < MIN_FIRST_PASS_ANNOTATIONS_FOR_SCORES:
        return False, None, first_pass_annotations
    if any(
        annotation.accuracy_type == IncorrectAnnotationType.NO_FACT
        for annotation in first_pass_annotations
    ):
        return False, None, first_pass_annotations

    return (
        True,
        _consensus_bool(says_has_error(annotation) for annotation in first_pass_annotations),
        first_pass_annotations,
    )


def _aggregate_judgment_consensus(
    judgments: Iterable[SectionAdjudicationModel | SectionFactCheckingModel],
) -> bool | None:
    """Return unanimous aggregate hallucination judgment, or None when judgments disagree."""
    return _consensus_bool(
        judgment.aggregate_judgment == AdjudicationJudgment.HALLUCINATION for judgment in judgments
    )


def _first_pass_score_gold_label(
    answer_key_name: str,
    *,
    first_pass_consensus: bool | None,
    adjudications: list[SectionAdjudicationModel],
    fact_checkings: list[SectionFactCheckingModel],
) -> bool | None:
    """Return the binary gold label for a first-pass annotator score row."""
    if answer_key_name == "first_pass_annotators":
        return first_pass_consensus

    if answer_key_name == "first_pass_annotators_plus_medical_experts":
        if adjudications:
            return _aggregate_judgment_consensus(adjudications)
        return first_pass_consensus

    if answer_key_name == "first_pass_annotators_plus_fact_checkers":
        if fact_checkings:
            return _aggregate_judgment_consensus(fact_checkings)
        return first_pass_consensus

    raise ValueError(f"Unexpected first-pass score answer key: {answer_key_name!r}")


def _compute_binary_prf1(
    golds: Iterable[bool],
    preds: Iterable[bool],
) -> tuple[float | None, float | None, float | None]:
    """Compute binary precision, recall, and F1."""
    gold_list = list(golds)
    pred_list = list(preds)
    if len(gold_list) != len(pred_list):
        raise ValueError("Gold and prediction lists must be the same length.")
    if not gold_list:
        return None, None, None

    true_positives = sum(gold and pred for gold, pred in zip(gold_list, pred_list, strict=True))
    false_positives = sum(
        (not gold) and pred for gold, pred in zip(gold_list, pred_list, strict=True)
    )
    false_negatives = sum(
        gold and (not pred) for gold, pred in zip(gold_list, pred_list, strict=True)
    )
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def first_pass_precision_recall_f1_by_type(
    annotation_data: list[EnhancedResponse],
    medical_expert_data: list[list[EnhancedResponse]],
    factchecking_data: list[list[EnhancedResponse]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, str | int | float | None]]:
    """Calculate first-pass annotator precision, recall, and F1 by annotator type."""
    adjudications_by_section = collate_adjudications(medical_expert_data)
    fact_checkings_by_section = collate_fact_checkings(factchecking_data)
    rows: list[dict[str, str | int | float | None]] = []

    for answer_key_name in FIRST_PASS_SCORE_ANSWER_KEYS:
        n_sections_with_answer_key = 0
        accumulators = {
            annotator_type: _ScoreAccumulator() for annotator_type in ANNOTATOR_TYPE_ORDER
        }
        for response in annotation_data:
            if response.annotations is None:
                continue
            for section in response.annotations.sections_with_annotations:
                scoreable, first_pass_consensus, first_pass_annotations = (
                    _first_pass_section_scoreability(section)
                )
                if not scoreable:
                    continue

                gold = _first_pass_score_gold_label(
                    answer_key_name,
                    first_pass_consensus=first_pass_consensus,
                    adjudications=adjudications_by_section.get(section.id, []),
                    fact_checkings=fact_checkings_by_section.get(section.id, []),
                )
                if gold is None:
                    continue
                n_sections_with_answer_key += 1

                for annotation in first_pass_annotations:
                    annotator_type = first_pass_annotator_type(annotation)
                    if annotator_type is None:
                        continue

                    accumulator = accumulators[annotator_type]
                    accumulator.golds.append(gold)
                    accumulator.preds.append(says_has_error(annotation))
                    accumulator.section_ids.append(section.id)
                    accumulator.response_ids.append(response.id)

        for annotator_type in ANNOTATOR_TYPE_ORDER:
            accumulator = accumulators[annotator_type]
            scoreable_responses = set(accumulator.response_ids)
            scoreable_sections = set(accumulator.section_ids)
            precision, recall, f1 = _compute_binary_prf1(accumulator.golds, accumulator.preds)
            response_id_to_item = {}
            for i, response_id in enumerate(accumulator.response_ids):
                response_data = response_id_to_item.setdefault(
                    response_id, {"golds": [], "preds": []}
                )
                response_data["golds"].append(accumulator.golds[i])
                response_data["preds"].append(accumulator.preds[i])
            response_ids_to_prf1: dict[
                tuple[str, ...], tuple[float | None, float | None, float | None]
            ] = {}

            def _lookup_or_compute_prf1_responselevel(
                response_ids: list[str],
                *,
                response_id_to_item=response_id_to_item,
                response_ids_to_prf1=response_ids_to_prf1,
            ) -> tuple[float | None, float | None, float | None]:
                key = tuple(response_ids)
                result = response_ids_to_prf1.get(key)
                if result is None:
                    golds, preds = [], []
                    for response_id in response_ids:
                        response_data = response_id_to_item.get(
                            response_id, {"golds": [], "preds": []}
                        )
                        golds.extend(response_data["golds"])
                        preds.extend(response_data["preds"])
                    result = response_ids_to_prf1[key] = _compute_binary_prf1(golds, preds)
                return result

            def _get_precision_responselevel(response_ids: list[str]) -> float | None:
                precision, _, _ = _lookup_or_compute_prf1_responselevel(response_ids)
                return precision

            def _get_recall_responselevel(response_ids: list[str]) -> float | None:
                _, recall, _ = _lookup_or_compute_prf1_responselevel(response_ids)
                return recall

            def _get_f1_responselevel(response_ids: list[str]) -> float | None:
                _, _, f1 = _lookup_or_compute_prf1_responselevel(response_ids)
                return f1

            response_level_cis = bootstrap_confidence_intervals(
                [response.id for response in annotation_data if response.id in scoreable_responses],
                bootstrap_samples=bootstrap_samples,
                rng=np.random.default_rng(bootstrap_seed),
                statistics={
                    "precision": _get_precision_responselevel,
                    "recall": _get_recall_responselevel,
                    "f1": _get_f1_responselevel,
                },
            )
            scoreable_sections_per_response = [
                len(
                    [
                        s
                        for s in response.annotations.sections_with_annotations
                        if s.id in scoreable_sections
                    ]
                )
                for response in annotation_data
            ]

            section_id_to_item = {}
            for i, section_id in enumerate(accumulator.section_ids):
                section_data = section_id_to_item.setdefault(section_id, {"golds": [], "preds": []})
                section_data["golds"].append(accumulator.golds[i])
                section_data["preds"].append(accumulator.preds[i])
            section_ids_to_prf1: dict[
                tuple[str, ...], tuple[float | None, float | None, float | None]
            ] = {}

            def _lookup_or_compute_prf1_sectionlevel(
                section_ids: list[str],
                *,
                section_id_to_item=section_id_to_item,
                section_ids_to_prf1=section_ids_to_prf1,
            ) -> tuple[float | None, float | None, float | None]:
                key = tuple(section_ids)
                result = section_ids_to_prf1.get(key)
                if result is None:
                    golds, preds = [], []
                    for section_id in section_ids:
                        section_data = section_id_to_item.get(
                            section_id, {"golds": [], "preds": []}
                        )
                        golds.extend(section_data["golds"])
                        preds.extend(section_data["preds"])
                    result = section_ids_to_prf1[key] = _compute_binary_prf1(golds, preds)
                return result

            def _get_precision_sectionlevel(response_ids: list[str]) -> float | None:
                precision, _, _ = _lookup_or_compute_prf1_sectionlevel(response_ids)
                return precision

            def _get_recall_sectionlevel(response_ids: list[str]) -> float | None:
                _, recall, _ = _lookup_or_compute_prf1_sectionlevel(response_ids)
                return recall

            def _get_f1_sectionlevel(response_ids: list[str]) -> float | None:
                _, _, f1 = _lookup_or_compute_prf1_sectionlevel(response_ids)
                return f1

            section_level_cis = bootstrap_confidence_intervals(
                [
                    section.id
                    for response in annotation_data
                    for section in response.annotations.sections_with_annotations
                    if section.id in scoreable_sections
                ],
                bootstrap_samples=bootstrap_samples,
                rng=np.random.default_rng(bootstrap_seed),
                statistics={
                    "precision": _get_precision_sectionlevel,
                    "recall": _get_recall_sectionlevel,
                    "f1": _get_f1_sectionlevel,
                },
            )
            ci_desc = format(BOOTSTRAP_CONFIDENCE_LEVEL, ".0%")
            rows.append(
                {
                    "answer_key": answer_key_name,
                    "annotator_type": annotator_type,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    f"precision_bootstrap{ci_desc}ci_responselevel": response_level_cis.get(
                        "precision_ci_95"
                    ),
                    f"recall_bootstrap{ci_desc}ci_responselevel": response_level_cis.get(
                        "recall_ci_95"
                    ),
                    f"f1_bootstrap{ci_desc}ci_responselevel": response_level_cis.get("f1_ci_95"),
                    f"precision_bootstrap{ci_desc}ci_sectionlevel": section_level_cis.get(
                        "precision_ci_95"
                    ),
                    f"recall_bootstrap{ci_desc}ci_sectionlevel": section_level_cis.get(
                        "recall_ci_95"
                    ),
                    f"f1_bootstrap{ci_desc}ci_sectionlevel": section_level_cis.get("f1_ci_95"),
                    "n_answer_key_sections": n_sections_with_answer_key,
                    "n_sections_with_predictions": len(accumulator.section_ids),
                    "n_prediction_rows": len(accumulator.preds),
                    "min_scoreable_sections_per_response": min(scoreable_sections_per_response),
                    "mean_scoreable_sections_per_response": np.mean(
                        scoreable_sections_per_response
                    ),
                    "max_scoreable_sections_per_response": max(scoreable_sections_per_response),
                }
            )

    return rows


STEP1_LABELS = {
    AdjudicationJudgment.MAJOR: "major",
    AdjudicationJudgment.MINOR: "minor",
    AdjudicationJudgment.NEEDS_RESEARCH: "needs_research",
    AdjudicationJudgment.NO_HALLUCINATION: "no_hallucination",
}

MEDICAL_EXPERT_STEP2_LABELS = {
    AdjudicationJudgment.WITH_COMMENTS_MAJOR: "major",
    AdjudicationJudgment.WITH_COMMENTS_MINOR: "minor",
    AdjudicationJudgment.WITH_COMMENTS_NO_HALLUCINATION: "no_hallucination",
}

FACT_CHECKER_STEP2_LABELS = {
    FactCheckingExcerptJudgment.EVIDENCE_SHOWS_INCORRECT_INFO_IN_EXCERPT: (
        "evidence_shows_error_in_excerpt"
    ),
    FactCheckingExcerptJudgment.LACK_OF_EVIDENCE_INDICATES_FABRICATION: (
        "lack_of_evidence_suggests_fabrication_in_excerpt"
    ),
    FactCheckingExcerptJudgment.EVIDENCE_FULLY_SUPPORTS_EXCERPT: (
        "evidence_shows_excerpt_fully_correct"
    ),
    FactCheckingExcerptJudgment.EVIDENCE_UNRELATED: "evidence_unrelated_to_excerpt",
}

FACT_CHECKER_STEP3_LABELS = {
    FactCheckingCommentsJudgment.EVIDENCE_SUPPORTS_IDENTIFIED_ERROR: (
        "evidence_shows_comments_correctly_identify_error"
    ),
    FactCheckingCommentsJudgment.EVIDENCE_SUPPORTS_IDENTIFIED_FABRICATION: (
        "lack_of_evidence_shows_comments_correctly_identify_fabrication"
    ),
    FactCheckingCommentsJudgment.EVIDENCE_CONTRADICTS_COMMENTS: "evidence_contradicts_comments",
    FactCheckingCommentsJudgment.EVIDENCE_UNRELATED: "evidence_unrelated_to_comments",
}


def section_level_statistics(
    annotation_data: list[EnhancedResponse],
) -> dict[str, int | float | None]:
    """Calculate section-level statistics."""
    sections = [
        section
        for response in annotation_data
        for section in response.annotations.sections_with_annotations
    ]
    response_has_injection = {
        response.id: any(
            a.annotator_id == INJECTOR_ID
            for section in response.annotations.sections_with_annotations
            for a in section.annotations
        )
        for response in annotation_data
    }
    section_is_from_injected = {
        section.id: response_has_injection[response.id]
        for response in annotation_data
        for section in response.annotations.sections_with_annotations
    }
    n_all_marked_claim = sum(
        all(
            annotation.accuracy_type != IncorrectAnnotationType.NO_FACT
            for annotation in section.annotations
        )
        for section in sections
    )
    n_sections_with_at_least_two_human_annotators = sum(
        all(
            annotation.accuracy_type != IncorrectAnnotationType.NO_FACT
            for annotation in section.annotations
        )
        and (sum(is_human(annotation) for annotation in section.annotations) >= 2)
        for section in sections
    )
    n_single_annotated = 0
    n_multiple_annotated = 0
    n_claimful_single_annotated = 0
    n_claimful_multiple_annotated = 0
    n_sections_where_all_agree = 0
    n_sections_where_all_humans_agree_and_at_least_two_human_annotators = 0
    n_sections_where_all_humans_agree_has_error_and_at_least_two_human_annotators = 0
    n_sections_where_humans_disagree = 0
    n_sections_where_some_disagree = 0
    n_all_agree_has_error = 0
    n_only_laj_has_error = 0
    n_only_injector_has_error = 0
    n_humans_say_error_while_injector_and_detector_say_no = 0
    n_humans_say_error_while_detector_says_no_excluding_responses_with_injected_hallucination = 0
    for section in sections:
        human_labels = [
            says_has_error(annotation) for annotation in section.annotations if is_human(annotation)
        ]
        if len(human_labels) == 1:
            n_single_annotated += 1
        elif len(human_labels) >= 2:
            n_multiple_annotated += 1
        if all(
            annotation.accuracy_type != IncorrectAnnotationType.NO_FACT
            for annotation in section.annotations
        ):
            if len(human_labels) == 1:
                n_claimful_single_annotated += 1
            elif len(human_labels) >= 2:
                n_claimful_multiple_annotated += 1
                if len(set(human_labels)) == 1:
                    n_sections_where_all_humans_agree_and_at_least_two_human_annotators += 1
                    n_sections_where_all_humans_agree_has_error_and_at_least_two_human_annotators += int(
                        human_labels[0]
                    )
                else:
                    n_sections_where_humans_disagree += 1
            label_set = {says_has_error(annotation) for annotation in section.annotations}
            if len(label_set) == 1:
                n_sections_where_all_agree += 1
                n_all_agree_has_error += int(True in label_set)
            else:
                n_sections_where_some_disagree += 1
                laj_says_has_error = any(
                    says_has_error(annotation)
                    for annotation in section.annotations
                    if annotation.annotator_id == LAJ_ID
                )
                injector_says_has_error = any(
                    says_has_error(annotation)
                    for annotation in section.annotations
                    if annotation.annotator_id == INJECTOR_ID
                )
                humans_say_no_error = all(
                    not says_has_error(annotation)
                    for annotation in section.annotations
                    if is_human(annotation)
                )
                n_only_laj_has_error += int(laj_says_has_error and humans_say_no_error)
                n_only_injector_has_error += int(
                    injector_says_has_error and humans_say_no_error and not laj_says_has_error
                )
                humans_say_error = all(
                    says_has_error(annotation)
                    for annotation in section.annotations
                    if is_human(annotation)
                )
                n_humans_say_error_while_injector_and_detector_say_no += int(
                    humans_say_error and (not laj_says_has_error and not injector_says_has_error)
                )
                n_humans_say_error_while_detector_says_no_excluding_responses_with_injected_hallucination += int(
                    humans_say_error
                    and not laj_says_has_error
                    and not section_is_from_injected[section.id]
                )

    return {
        "n_sections": len(sections),
        "n_sections_single_annotated": n_single_annotated,
        "n_sections_multiple_annotated": n_multiple_annotated,
        "n_claimful_sections_single_annotated": n_claimful_single_annotated,
        "n_claimful_sections_multiple_annotated": n_claimful_multiple_annotated,
        "n_sections_all_ann_marked_claim": n_all_marked_claim,
        "pct_sections_all_ann_marked_claim": 100 * n_all_marked_claim / len(sections),
        "human_ann_no_claim_agreement": calculate_no_claim_krippendorff(sections),
        "human_ann_has_error_agreement": calculate_has_error_krippendorff(
            sections, check_is_human=True
        ),
        "all_ann_has_error_agreement": calculate_has_error_krippendorff(
            sections, check_is_human=False
        ),
        "n_claimful_sections_with_at_least_two_human_annotators": n_sections_with_at_least_two_human_annotators,
        "n_claimful_sections_with_at_least_two_human_annotators_where_all_humans_agree": n_sections_where_all_humans_agree_and_at_least_two_human_annotators,
        "n_claimful_sections_with_at_least_two_human_annotators_where_all_humans_agree_has_error": n_sections_where_all_humans_agree_has_error_and_at_least_two_human_annotators,
        "n_claimful_sections_where_humans_disagree": n_sections_where_humans_disagree,
        "n_claimful_sections_all_agree_has_error_or_not": n_sections_where_all_agree,
        "pct_has_error_among_claimful_section_all_agree": n_all_agree_has_error
        / n_sections_where_all_agree,
        "n_laj_only_has_error": n_only_laj_has_error,
        "pct_laj_only_has_error_among_claimful_section_where_claimful": n_only_laj_has_error
        / n_all_marked_claim,
        "n_injector_only_has_error": n_only_injector_has_error,
        "n_humans_say_error_while_injector_and_detector_say_no": n_humans_say_error_while_injector_and_detector_say_no,
        "n_humans_say_error_while_detector_says_no_excluding_responses_with_injected_hallucination": n_humans_say_error_while_detector_says_no_excluding_responses_with_injected_hallucination,
    }


def _first_pass_annotator_judgments_per_claimful_section(
    sections: list[SectionResponseAnnotation],
) -> list[int]:
    """Count first-pass has-error judgments used in agreement calculations."""
    judgments_per_section = []
    for section in sections:
        if any(
            is_human(annotation) and annotation.accuracy_type == IncorrectAnnotationType.NO_FACT
            for annotation in section.annotations
        ):
            continue

        n_judgments = sum(is_human(annotation) for annotation in section.annotations)
        judgments_per_section.append(n_judgments)

    return judgments_per_section


def _fill_missing_items(
    annotators2items: dict[str, dict[str, int | None]],
) -> dict[str, dict[str, int | None]]:
    """Fill missing item judgments with None, preserving a consistent item order."""
    item_ids = sorted({item_id for items in annotators2items.values() for item_id in items})
    return {
        annotator_id: {item_id: items.get(item_id) for item_id in item_ids}
        for annotator_id, items in sorted(annotators2items.items())
    }


def _first_pass_summary_dataset(
    sections: list[SectionResponseAnnotation],
) -> dict[str, dict[bool, int]]:
    """Build first-pass human annotator judgments as an xRR summary dataset."""
    dataset: dict[str, dict[bool, int]] = {}
    for section in sections:
        if any(
            is_human(annotation1) and annotation1.accuracy_type == IncorrectAnnotationType.NO_FACT
            for annotation1 in section.annotations
        ):
            continue
        for annotation in section.annotations:
            if is_human(annotation):
                judgment = says_has_error(annotation)
                judgments = dataset.setdefault(section.id, {})
                judgments[judgment] = judgments.get(judgment, 0) + 1
    return dataset


def _laj_summary_dataset(sections: list[SectionResponseAnnotation]) -> dict[str, dict[bool, int]]:
    """Build LAJ judgments as an xRR summary dataset."""
    dataset: dict[str, dict[bool, int]] = {}
    for section in sections:
        if any(
            is_human(annotation1) and annotation1.accuracy_type == IncorrectAnnotationType.NO_FACT
            for annotation1 in section.annotations
        ):
            continue
        for annotation in section.annotations:
            if annotation.annotator_id == LAJ_ID:
                judgment = says_has_error(annotation)
                judgments = dataset.setdefault(section.id, {})
                judgments[judgment] = judgments.get(judgment, 0) + 1
    return dataset


def _medical_expert_summary_dataset(
    medical_expert_data: list[list[EnhancedResponse]],
) -> dict[str, dict[bool, int]]:
    """Build medical expert aggregate judgments as an xRR summary dataset."""
    dataset: dict[str, dict[bool, int]] = {}
    for section_id, adjudications in collate_adjudications(medical_expert_data).items():
        for adjudication in adjudications:
            judgment = adjudication.aggregate_judgment == AdjudicationJudgment.HALLUCINATION
            judgments = dataset.setdefault(section_id, {})
            judgments[judgment] = judgments.get(judgment, 0) + 1
    return dataset


def _factchecker_summary_dataset(
    factchecking_data: list[list[EnhancedResponse]],
) -> dict[str, dict[bool, int]]:
    """Build fact-checker aggregate judgments as an xRR summary dataset."""
    dataset: dict[str, dict[bool, int]] = {}
    for section_id, fact_checkings in collate_fact_checkings(factchecking_data).items():
        for fact_checking in fact_checkings:
            judgment = fact_checking.aggregate_judgment == AdjudicationJudgment.HALLUCINATION
            judgments = dataset.setdefault(section_id, {})
            judgments[judgment] = judgments.get(judgment, 0) + 1
    return dataset


def _summary_dataset_count_judgments(dataset: dict[str, dict[int, int]]) -> int:
    """Count judgments in an xRR summary dataset."""
    return sum(sum(judgment_counts.values()) for judgment_counts in dataset.values())


def _cross_pool_agreement_statistics(
    dataset_x: dict[str, dict[int, int]], dataset_y: dict[str, dict[int, int]]
) -> dict[str, int | float | tuple[int, int] | None]:
    """Calculate xRR agreement statistics across two summary datasets."""
    section_ids = sorted(dataset_x.keys() & dataset_y.keys())
    if not section_ids:
        return {
            "count_judgments": 0,
            "judgments_per_section_range": None,
            "agreement": None,
        }

    intersected_dataset_x = {section_id: dataset_x[section_id] for section_id in section_ids}
    intersected_dataset_y = {section_id: dataset_y[section_id] for section_id in section_ids}
    judgments_per_section = [
        sum(intersected_dataset_x[section_id].values())
        + sum(intersected_dataset_y[section_id].values())
        for section_id in section_ids
    ]
    xrr_agreement = compute_xrr_with_summary_datasets(
        intersected_dataset_x,
        intersected_dataset_y,
        DistanceFunctions.nominal,
        XrrMetrics.WITH_MISSING_DATA,
    )
    return {
        "count_sections_with_pairable_judgments": len(section_ids),
        "count_judgments": _summary_dataset_count_judgments(intersected_dataset_x)
        + _summary_dataset_count_judgments(intersected_dataset_y),
        "count_true_judgments": sum(
            item_judgments.get(True, 0)
            for item, item_judgments in intersected_dataset_x.items()
            if sum(intersected_dataset_x[item].values()) >= 1 and sum(item_judgments.values()) >= 1
        )
        + sum(
            item_judgments.get(True, 0)
            for item, item_judgments in intersected_dataset_y.items()
            if sum(intersected_dataset_x[item].values()) >= 1 and sum(item_judgments.values()) >= 1
        ),
        "count_false_judgments": sum(
            item_judgments.get(False, 0)
            for item, item_judgments in intersected_dataset_x.items()
            if sum(intersected_dataset_y[item].values()) >= 1 and sum(item_judgments.values()) >= 1
        )
        + sum(
            item_judgments.get(False, 0)
            for item, item_judgments in intersected_dataset_y.items()
            if sum(intersected_dataset_x[item].values()) >= 1 and sum(item_judgments.values()) >= 1
        ),
        "percent_all_agree": calculate_percent_all_agree_cross_pool(
            intersected_dataset_x, intersected_dataset_y
        ),
        "percent_agreement_macro": calculate_percent_agreement_cross_pool_macro_avg(
            intersected_dataset_x, intersected_dataset_y
        ),
        "percent_agreement_micro": calculate_percent_agreement_cross_pool_micro_avg(
            intersected_dataset_x, intersected_dataset_y
        ),
        "judgments_per_section_range": (min(judgments_per_section), max(judgments_per_section)),
        "cross_replication_reliability": xrr_agreement,
    }


ItemT = TypeVar("ItemT", bound=Hashable)
AnnotationT = TypeVar("AnnotationT", bound=Hashable)


def _single_pool_aggregate_judgment_statistics(
    annotators2sections: dict[str, dict[str, int | None]],
    summary_dataset: dict[ItemT, dict[AnnotationT, int]],
) -> dict[str, (int | float | tuple[int, int] | None)]:
    """Calculate single-pool agreement statistics from aggregate judgments."""
    section_ids = sorted(
        {section_id for sections in annotators2sections.values() for section_id in sections}
    )
    judgments_per_section = [
        sum(
            annotators2sections[annotator_id].get(section_id) is not None
            for annotator_id in annotators2sections
        )
        for section_id in section_ids
    ]
    agreement = calculate_krippendorff(_fill_missing_items(annotators2sections), level="nominal")

    judgments_per_section_range = min([ct for ct in judgments_per_section if ct >= 2]), max(
        [ct for ct in judgments_per_section if ct >= 2]
    )
    return {
        "count_sections_with_pairable_judgments": sum(1 for ct in judgments_per_section if ct >= 2),
        "count_judgments": sum(judgments_per_section),
        "count_true_judgments": sum(
            item_judgments.get(True, 0)
            for item_judgments in summary_dataset.values()
            if sum(item_judgments.values()) >= 2
        ),
        "count_false_judgments": sum(
            item_judgments.get(False, 0)
            for item_judgments in summary_dataset.values()
            if sum(item_judgments.values()) >= 2
        ),
        "percent_all_agree": calculate_percent_all_agree_same_pool(summary_dataset),
        "percent_agreement_macro": calculate_percent_agreement_same_pool_macro_avg(summary_dataset),
        "percent_agreement_micro": calculate_percent_agreement_same_pool_micro_avg(summary_dataset),
        "judgments_per_section_range": judgments_per_section_range,
        "krippendorffs_alpha": agreement,
    }


def _medical_expert_agreement_statistics(
    medical_expert_data: list[list[EnhancedResponse]],
    summary_dataset: dict[ItemT, dict[AnnotationT, int]],
) -> dict[str, (int | float | tuple[int, int] | None)]:
    """Calculate single-pool medical expert agreement statistics."""
    annotators2sections: dict[str, dict[str, int | None]] = {}
    for section_id, adjudications in collate_adjudications(medical_expert_data).items():
        for adjudication in adjudications:
            sections2values = annotators2sections.setdefault(adjudication.annotator_id, {})
            if section_id in sections2values:
                raise ValueError(
                    "Duplicate medical expert aggregate judgment for annotator "
                    f"{adjudication.annotator_id!r} on section {section_id!r}."
                )
            sections2values[section_id] = int(
                adjudication.aggregate_judgment == AdjudicationJudgment.HALLUCINATION
            )
    return _single_pool_aggregate_judgment_statistics(annotators2sections, summary_dataset)


def _factchecker_agreement_statistics(
    factchecking_data: list[list[EnhancedResponse]],
    summary_dataset: dict[ItemT, dict[AnnotationT, int]],
) -> dict[str, (int | float | tuple[int, int] | None)]:
    """Calculate single-pool fact-checker agreement statistics."""
    annotators2sections: dict[str, dict[str, int | None]] = {}
    for section_id, fact_checkings in collate_fact_checkings(factchecking_data).items():
        for fact_checking in fact_checkings:
            sections2values = annotators2sections.setdefault(fact_checking.annotator_id, {})
            if section_id in sections2values:
                raise ValueError(
                    "Duplicate fact-checker aggregate judgment for annotator "
                    f"{fact_checking.annotator_id!r} on section {section_id!r}."
                )
            sections2values[section_id] = int(
                fact_checking.aggregate_judgment == AdjudicationJudgment.HALLUCINATION
            )
    return _single_pool_aggregate_judgment_statistics(annotators2sections, summary_dataset)


def _calculate_pair_agreement_single_pool(
    summary_dataset: dict[ItemT, dict[AnnotationT, int]],
) -> tuple[list[int], list[int]]:
    pair_agreements = []
    pair_counts = []
    for ratings in summary_dataset.values():
        if sum(ratings.values()) < 2:
            continue
        pair_agreements.append(sum(math.comb(count, 2) for rating, count in ratings.items()))
        pair_counts.append(math.comb(sum(ratings.values()), 2))

    return pair_agreements, pair_counts


def calculate_percent_all_agree_same_pool(
    summary_dataset: dict[ItemT, dict[AnnotationT, int]],
) -> float:
    """Calculate percent all agree for a single pool of annotators, macro-averaging over items."""
    pair_agreements, pair_counts = _calculate_pair_agreement_single_pool(summary_dataset)
    return (
        100.0
        * sum(
            agreement == count
            for agreement, count in zip(pair_agreements, pair_counts, strict=True)
            if count
        )
        / len(pair_agreements)
    )


def calculate_percent_agreement_same_pool_macro_avg(
    summary_dataset: dict[ItemT, dict[AnnotationT, int]],
) -> float:
    """Calculate percent agreement for a single pool of annotators, macro-averaging over items."""
    pair_agreements, pair_counts = _calculate_pair_agreement_single_pool(summary_dataset)
    return (
        100.0
        * sum(
            agreement / count
            for agreement, count in zip(pair_agreements, pair_counts, strict=True)
            if count
        )
        / len(pair_agreements)
    )


def calculate_percent_agreement_same_pool_micro_avg(
    summary_dataset: dict[ItemT, dict[AnnotationT, int]],
) -> float:
    """Calculate percent agreement for a single pool of annotators, micro-averaging over items."""
    pair_agreements, pair_counts = _calculate_pair_agreement_single_pool(summary_dataset)
    return 100.0 * sum(pair_agreements) / sum(pair_counts)


def _calculate_pair_agreement_cross_pool(
    summary_dataset1: dict[ItemT, dict[AnnotationT, int]],
    summary_dataset2: dict[ItemT, dict[AnnotationT, int]],
) -> tuple[list[int], list[int]]:
    pair_agreements = []
    pair_counts = []
    intersection = sorted(summary_dataset1.keys() & summary_dataset2.keys())
    for item in intersection:
        ratings = sorted(summary_dataset1[item].keys() | summary_dataset2[item].keys())
        pair_agreements.append(
            sum(
                summary_dataset1[item].get(rating, 0) * summary_dataset2[item].get(rating, 0)
                for rating in ratings
            )
        )
        pair_counts.append(
            sum(summary_dataset1[item].values()) * sum(summary_dataset2[item].values())
        )

    return pair_agreements, pair_counts


def calculate_percent_all_agree_cross_pool(
    summary_dataset1: dict[ItemT, dict[AnnotationT, int]],
    summary_dataset2: dict[ItemT, dict[AnnotationT, int]],
) -> float:
    """Calculate percent all agree across pools of annotators, macro-averaging over items."""
    pair_agreements, pair_counts = _calculate_pair_agreement_cross_pool(
        summary_dataset1, summary_dataset2
    )
    return (
        100.0
        * sum(
            agreement == count
            for agreement, count in zip(pair_agreements, pair_counts, strict=True)
            if count
        )
        / len(pair_agreements)
    )


def calculate_percent_agreement_cross_pool_macro_avg(
    summary_dataset1: dict[ItemT, dict[AnnotationT, int]],
    summary_dataset2: dict[ItemT, dict[AnnotationT, int]],
) -> float:
    """Calculate percent agreement across pools of annotators, macro-averaging over items."""
    pair_agreements, pair_counts = _calculate_pair_agreement_cross_pool(
        summary_dataset1, summary_dataset2
    )
    return (
        100.0
        * sum(
            agreement / count
            for agreement, count in zip(pair_agreements, pair_counts, strict=True)
            if count
        )
        / len(pair_agreements)
    )


def calculate_percent_agreement_cross_pool_micro_avg(
    summary_dataset1: dict[ItemT, dict[AnnotationT, int]],
    summary_dataset2: dict[ItemT, dict[AnnotationT, int]],
) -> float:
    """Calculate percent agreement across pools of annotators, micro-averaging over items."""
    pair_agreements, pair_counts = _calculate_pair_agreement_cross_pool(
        summary_dataset1, summary_dataset2
    )
    return 100.0 * sum(pair_agreements) / sum(pair_counts)


def _resample_summary_dataset(
    summary_dataset: dict[str, dict[bool, int]], sampled_item_ids: list[str]
) -> dict[str, dict[bool, int]]:
    """Resample a summary dataset, keeping duplicate sampled sections as distinct items."""
    return {
        f"{sample_index}:{item_id}": dict(summary_dataset[item_id])
        for sample_index, item_id in enumerate(sampled_item_ids)
    }


def _resample_annotators2items(
    annotators2items: dict[str, dict[str, int | None]], sampled_item_ids: list[str]
) -> dict[str, dict[str, int | None]]:
    """Resample an annotator-item mapping, keeping duplicate sampled items as distinct items."""
    return {
        annotator_id: {
            f"{sample_index}:{item_id}": items.get(item_id)
            for sample_index, item_id in enumerate(sampled_item_ids)
        }
        for annotator_id, items in annotators2items.items()
    }


def _percentile_confidence_interval(values: list[float]) -> tuple[float, float] | None:
    """Return a percentile 95% confidence interval from bootstrap values."""
    if not values:
        return None

    alpha = 1.0 - BOOTSTRAP_CONFIDENCE_LEVEL
    lower_percentile = 100.0 * alpha / 2.0
    upper_percentile = 100.0 * (1.0 - alpha / 2.0)
    lower, upper = np.percentile(values, [lower_percentile, upper_percentile])
    return float(lower), float(upper)


def bootstrap_confidence_intervals(
    item_ids: list[str],
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
    statistics: dict[str, Callable[[list[str]], float | None]],
) -> dict[str, tuple[float, float] | None]:
    """Calculate percentile bootstrap confidence intervals for statistics over items."""
    if bootstrap_samples <= 0:
        return {}
    if not item_ids:
        return {f"{name}_ci_95": None for name in statistics}

    values_by_statistic: dict[str, list[float]] = {name: [] for name in statistics}
    for _ in range(bootstrap_samples):
        sampled_item_ids = [item_ids[int(rng.integers(0, len(item_ids)))] for _item_id in item_ids]
        for name, statistic in statistics.items():
            try:
                value = statistic(sampled_item_ids)
            except (ArithmeticError, ValueError):
                continue

            if value is not None and math.isfinite(value):
                values_by_statistic[name].append(float(value))

    return {
        f"{name}_ci_95": _percentile_confidence_interval(values)
        for name, values in values_by_statistic.items()
    }


def _compute_xrr_nominal_with_summary_datasets(
    dataset_x: dict[str, dict[bool, int]], dataset_y: dict[str, dict[bool, int]]
) -> float | None:
    """Compute nominal xRR with the same formula as the xRR helper, optimized for bootstrap."""
    item_ids = list(dataset_x.keys() & dataset_y.keys())
    if not item_ids:
        return None

    total_annotations_x = sum(sum(dataset_x[item_id].values()) for item_id in item_ids)
    total_annotations_y = sum(sum(dataset_y[item_id].values()) for item_id in item_ids)
    if total_annotations_x == 0 or total_annotations_y == 0:
        return None

    observed_disagreement = 0.0
    marginal_counts_x: dict[bool, int] = {}
    marginal_counts_y: dict[bool, int] = {}
    for item_id in item_ids:
        item_counts_x = dataset_x[item_id]
        item_counts_y = dataset_y[item_id]
        item_total_x = sum(item_counts_x.values())
        item_total_y = sum(item_counts_y.values())
        if item_total_x == 0 or item_total_y == 0:
            continue

        item_disagreement = sum(
            count_x * count_y
            for label_x, count_x in item_counts_x.items()
            for label_y, count_y in item_counts_y.items()
            if label_x != label_y
        )
        observed_disagreement += (
            item_disagreement * (item_total_x + item_total_y) / (item_total_x * item_total_y)
        )

        for label, count in item_counts_x.items():
            marginal_counts_x[label] = marginal_counts_x.get(label, 0) + count
        for label, count in item_counts_y.items():
            marginal_counts_y[label] = marginal_counts_y.get(label, 0) + count

    observed_disagreement /= total_annotations_x + total_annotations_y
    expected_disagreement = sum(
        count_x * count_y
        for label_x, count_x in marginal_counts_x.items()
        for label_y, count_y in marginal_counts_y.items()
        if label_x != label_y
    ) / (total_annotations_x * total_annotations_y)
    if expected_disagreement == 0:
        return None

    return 1.0 - observed_disagreement / expected_disagreement


def _single_pool_agreement_confidence_intervals(
    summary_dataset: dict[str, dict[bool, int]],
    annotators2items: dict[str, dict[str, int | None]],
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float] | None]:
    """Calculate bootstrap confidence intervals for same-pool agreement statistics."""
    item_ids = list(summary_dataset)

    def resampled_summary(sampled_item_ids: list[str]) -> dict[str, dict[bool, int]]:
        return _resample_summary_dataset(summary_dataset, sampled_item_ids)

    def resampled_annotators2items(
        sampled_item_ids: list[str],
    ) -> dict[str, dict[str, int | None]]:
        return _resample_annotators2items(annotators2items, sampled_item_ids)

    return bootstrap_confidence_intervals(
        item_ids,
        bootstrap_samples=bootstrap_samples,
        rng=rng,
        statistics={
            "percent_all_agree": lambda sampled_item_ids: calculate_percent_all_agree_same_pool(
                resampled_summary(sampled_item_ids)
            ),
            "percent_agreement_macro": (
                lambda sampled_item_ids: calculate_percent_agreement_same_pool_macro_avg(
                    resampled_summary(sampled_item_ids)
                )
            ),
            "percent_agreement_micro": (
                lambda sampled_item_ids: calculate_percent_agreement_same_pool_micro_avg(
                    resampled_summary(sampled_item_ids)
                )
            ),
            "krippendorffs_alpha": lambda sampled_item_ids: calculate_krippendorff(
                resampled_annotators2items(sampled_item_ids), level="nominal"
            ),
        },
    )


def _cross_pool_agreement_confidence_intervals(
    dataset_x: dict[str, dict[bool, int]],
    dataset_y: dict[str, dict[bool, int]],
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float] | None]:
    """Calculate bootstrap confidence intervals for cross-pool agreement statistics."""
    item_ids = list(dataset_x.keys() & dataset_y.keys())

    def resampled_datasets(
        sampled_item_ids: list[str],
    ) -> tuple[dict[str, dict[bool, int]], dict[str, dict[bool, int]]]:
        return (
            _resample_summary_dataset(dataset_x, sampled_item_ids),
            _resample_summary_dataset(dataset_y, sampled_item_ids),
        )

    def percent_all_agree(sampled_item_ids: list[str]) -> float:
        resampled_x, resampled_y = resampled_datasets(sampled_item_ids)
        return calculate_percent_all_agree_cross_pool(resampled_x, resampled_y)

    def percent_agreement_macro(sampled_item_ids: list[str]) -> float:
        resampled_x, resampled_y = resampled_datasets(sampled_item_ids)
        return calculate_percent_agreement_cross_pool_macro_avg(resampled_x, resampled_y)

    def percent_agreement_micro(sampled_item_ids: list[str]) -> float:
        resampled_x, resampled_y = resampled_datasets(sampled_item_ids)
        return calculate_percent_agreement_cross_pool_micro_avg(resampled_x, resampled_y)

    def cross_replication_reliability(sampled_item_ids: list[str]) -> float | None:
        resampled_x, resampled_y = resampled_datasets(sampled_item_ids)
        return _compute_xrr_nominal_with_summary_datasets(resampled_x, resampled_y)

    return bootstrap_confidence_intervals(
        item_ids,
        bootstrap_samples=bootstrap_samples,
        rng=rng,
        statistics={
            "percent_all_agree": percent_all_agree,
            "percent_agreement_macro": percent_agreement_macro,
            "percent_agreement_micro": percent_agreement_micro,
            "cross_replication_reliability": cross_replication_reliability,
        },
    )


def _first_pass_has_error_annotators2sections(
    sections: list[SectionResponseAnnotation],
) -> dict[str, dict[str, int | None]]:
    """Build first-pass has-error judgments by annotator and section."""
    annotators2sections: dict[str, dict[str, int | None]] = {}
    for section in sections:
        if any(
            is_human(annotation) and annotation.accuracy_type == IncorrectAnnotationType.NO_FACT
            for annotation in section.annotations
        ):
            continue

        for annotation in section.annotations:
            if is_human(annotation):
                sections2values = annotators2sections.setdefault(annotation.annotator_id, {})
                sections2values[section.id] = int(says_has_error(annotation))

    return _fill_missing_items(annotators2sections)


def _medical_expert_annotators2sections(
    medical_expert_data: list[list[EnhancedResponse]],
) -> dict[str, dict[str, int | None]]:
    """Build medical expert aggregate judgments by annotator and section."""
    annotators2sections: dict[str, dict[str, int | None]] = {}
    for section_id, adjudications in collate_adjudications(medical_expert_data).items():
        for adjudication in adjudications:
            sections2values = annotators2sections.setdefault(adjudication.annotator_id, {})
            if section_id in sections2values:
                raise ValueError(
                    "Duplicate medical expert aggregate judgment for annotator "
                    f"{adjudication.annotator_id!r} on section {section_id!r}."
                )
            sections2values[section_id] = int(
                adjudication.aggregate_judgment == AdjudicationJudgment.HALLUCINATION
            )
    return _fill_missing_items(annotators2sections)


def _factchecker_annotators2sections(
    factchecking_data: list[list[EnhancedResponse]],
) -> dict[str, dict[str, int | None]]:
    """Build fact-checker aggregate judgments by annotator and section."""
    annotators2sections: dict[str, dict[str, int | None]] = {}
    for section_id, fact_checkings in collate_fact_checkings(factchecking_data).items():
        for fact_checking in fact_checkings:
            sections2values = annotators2sections.setdefault(fact_checking.annotator_id, {})
            if section_id in sections2values:
                raise ValueError(
                    "Duplicate fact-checker aggregate judgment for annotator "
                    f"{fact_checking.annotator_id!r} on section {section_id!r}."
                )
            sections2values[section_id] = int(
                fact_checking.aggregate_judgment == AdjudicationJudgment.HALLUCINATION
            )
    return _fill_missing_items(annotators2sections)


def agreement_statistics(
    annotation_data: list[EnhancedResponse],
    medical_expert_data: list[list[EnhancedResponse]],
    factchecking_data: list[list[EnhancedResponse]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, dict[str, int | float | tuple[int | float, int | float] | None]]:
    """Calculate agreement within and across approaches."""
    rng = np.random.default_rng(bootstrap_seed)
    sections = [
        section
        for response in annotation_data
        for section in response.annotations.sections_with_annotations
    ]
    first_pass_judgments_per_section = _first_pass_annotator_judgments_per_claimful_section(
        sections
    )
    # Construct datasets to use for xRR calculations
    first_pass_dataset = _first_pass_summary_dataset(sections)
    laj_dataset = _laj_summary_dataset(sections)
    medical_expert_dataset = _medical_expert_summary_dataset(medical_expert_data)
    factchecker_dataset = _factchecker_summary_dataset(factchecking_data)

    first_pass_annotators2sections = _first_pass_has_error_annotators2sections(sections)
    medical_expert_annotators2sections = _medical_expert_annotators2sections(medical_expert_data)
    factchecker_annotators2sections = _factchecker_annotators2sections(factchecking_data)

    first_pass_stats = {
        "count_sections_with_pairable_judgments": sum(
            1 for ct in first_pass_judgments_per_section if ct >= 2
        ),
        "count_judgments": sum(first_pass_judgments_per_section),
        "count_true_judgments": sum(
            item_judgments.get(True, 0)
            for item_judgments in first_pass_dataset.values()
            if sum(item_judgments.values()) >= 2
        ),
        "count_false_judgments": sum(
            item_judgments.get(False, 0)
            for item_judgments in first_pass_dataset.values()
            if sum(item_judgments.values()) >= 2
        ),
        "percent_all_agree": calculate_percent_all_agree_same_pool(first_pass_dataset),
        "percent_agreement_macro": calculate_percent_agreement_same_pool_macro_avg(
            first_pass_dataset
        ),
        "percent_agreement_micro": calculate_percent_agreement_same_pool_micro_avg(
            first_pass_dataset
        ),
        "judgments_per_section_range": (
            min([ct for ct in first_pass_judgments_per_section if ct >= 2]),
            max([ct for ct in first_pass_judgments_per_section if ct >= 2]),
        ),
        "krippendorffs_alpha": calculate_has_error_krippendorff(sections, check_is_human=True),
    }
    first_pass_stats.update(
        _single_pool_agreement_confidence_intervals(
            first_pass_dataset,
            first_pass_annotators2sections,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        )
    )

    medical_expert_stats = _single_pool_aggregate_judgment_statistics(
        medical_expert_annotators2sections, medical_expert_dataset
    )
    medical_expert_stats.update(
        _single_pool_agreement_confidence_intervals(
            medical_expert_dataset,
            medical_expert_annotators2sections,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        )
    )

    factchecker_stats = _single_pool_aggregate_judgment_statistics(
        factchecker_annotators2sections, factchecker_dataset
    )
    factchecker_stats.update(
        _single_pool_agreement_confidence_intervals(
            factchecker_dataset,
            factchecker_annotators2sections,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        )
    )

    first_pass_vs_medical_expert_stats = _cross_pool_agreement_statistics(
        first_pass_dataset, medical_expert_dataset
    )
    first_pass_vs_medical_expert_stats.update(
        _cross_pool_agreement_confidence_intervals(
            first_pass_dataset,
            medical_expert_dataset,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        )
    )

    first_pass_vs_factchecker_stats = _cross_pool_agreement_statistics(
        first_pass_dataset, factchecker_dataset
    )
    first_pass_vs_factchecker_stats.update(
        _cross_pool_agreement_confidence_intervals(
            first_pass_dataset,
            factchecker_dataset,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        )
    )

    laj_vs_medical_expert_stats = _cross_pool_agreement_statistics(
        laj_dataset, medical_expert_dataset
    )
    laj_vs_medical_expert_stats.update(
        _cross_pool_agreement_confidence_intervals(
            laj_dataset,
            medical_expert_dataset,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        )
    )

    laj_vs_factchecker_stats = _cross_pool_agreement_statistics(laj_dataset, factchecker_dataset)
    laj_vs_factchecker_stats.update(
        _cross_pool_agreement_confidence_intervals(
            laj_dataset,
            factchecker_dataset,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        )
    )

    medical_expert_vs_factchecker_stats = _cross_pool_agreement_statistics(
        medical_expert_dataset, factchecker_dataset
    )
    medical_expert_vs_factchecker_stats.update(
        _cross_pool_agreement_confidence_intervals(
            medical_expert_dataset,
            factchecker_dataset,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        )
    )

    return {
        "first_pass_annotators": first_pass_stats,
        "medical_experts": medical_expert_stats,
        "factcheckers": factchecker_stats,
        "first_pass_annotators_vs_medical_experts": first_pass_vs_medical_expert_stats,
        "first_pass_annotators_vs_factcheckers": first_pass_vs_factchecker_stats,
        "laj_vs_medical_experts": laj_vs_medical_expert_stats,
        "laj_vs_factcheckers": laj_vs_factchecker_stats,
        "medical_experts_vs_factcheckers": medical_expert_vs_factchecker_stats,
    }


S = TypeVar("S")
T = TypeVar("T")


def _empty_confusion_matrix(row_labels: list[S], column_labels: list[T]) -> dict[S, dict[T, int]]:
    """Create a confusion matrix initialized with zeroes."""
    return {row_label: dict.fromkeys(column_labels, 0) for row_label in row_labels}


def _increment_confusion_matrix(
    matrix: dict[str, dict[str, int | float | None]], row_label: str, column_label: str
) -> None:
    """Increment a single confusion matrix cell."""
    if row_label not in matrix:
        raise ValueError(f"Unexpected confusion matrix row: {row_label!r}")
    if column_label not in matrix[row_label]:
        raise ValueError(
            f"Unexpected confusion matrix column {column_label!r} for row {row_label!r}"
        )

    value = matrix[row_label][column_label]
    if value is None:
        raise ValueError(
            f"Cannot increment unset confusion matrix cell ({row_label!r}, {column_label!r})"
        )
    matrix[row_label][column_label] = value + 1


def confusion_matrices(
    medical_expert_data: list[list[EnhancedResponse]],
    factchecking_data: list[list[EnhancedResponse]],
) -> dict[str, dict[str, dict[str, int | None]]]:
    """Calculate medical expert and factchecker confusion matrices."""
    medical_experts_step1_vs_step2: dict[str, dict[str, int | None]] = {
        "major": {"total": 0},
        "minor": {"total": 0},
        "needs_research": {"major": 0, "minor": 0, "no_hallucination": 0},
        "no_hallucination": {"major": 0, "minor": 0, "no_hallucination": 0},
    }
    fact_checkers_step1_vs_step2 = _empty_confusion_matrix(
        list(STEP1_LABELS.values()), list(FACT_CHECKER_STEP2_LABELS.values())
    )
    fact_checkers_step1_vs_step3 = _empty_confusion_matrix(
        list(STEP1_LABELS.values()), list(FACT_CHECKER_STEP3_LABELS.values())
    )
    fact_checkers_step2_vs_step3 = _empty_confusion_matrix(
        list(FACT_CHECKER_STEP2_LABELS.values()), list(FACT_CHECKER_STEP3_LABELS.values())
    )

    for responses in medical_expert_data:
        for response in responses:
            assert response.annotations is not None
            for section in response.annotations.sections_with_annotations:
                adjudication = section.adjudication
                if adjudication is None:
                    continue

                step1_label = STEP1_LABELS[adjudication.q1_response]
                if adjudication.q1_response in {
                    AdjudicationJudgment.MAJOR,
                    AdjudicationJudgment.MINOR,
                }:
                    if adjudication.q3_response != AdjudicationJudgment.NOT_SET:
                        raise ValueError(
                            "Medical expert q3_response should be not_set when q1_response is "
                            f"{adjudication.q1_response!r}. Got {adjudication.q3_response!r}."
                        )
                    step2_label = "total"
                else:
                    step2_label = MEDICAL_EXPERT_STEP2_LABELS[adjudication.q3_response]
                _increment_confusion_matrix(
                    medical_experts_step1_vs_step2, step1_label, step2_label
                )

    for responses in factchecking_data:
        for response in responses:
            assert response.annotations is not None
            for section in response.annotations.sections_with_annotations:
                fact_checking = section.fact_checking
                if fact_checking is None:
                    continue

                step1_label = STEP1_LABELS[fact_checking.q1_response]
                step2_label = FACT_CHECKER_STEP2_LABELS[fact_checking.q5a_choice]
                step3_label = FACT_CHECKER_STEP3_LABELS[fact_checking.q5b_choice]

                _increment_confusion_matrix(fact_checkers_step1_vs_step2, step1_label, step2_label)
                _increment_confusion_matrix(fact_checkers_step1_vs_step3, step1_label, step3_label)
                _increment_confusion_matrix(fact_checkers_step2_vs_step3, step2_label, step3_label)

    return {
        "medical_experts_step1_vs_step2": medical_experts_step1_vs_step2,
        "fact_checkers_step1_vs_step2": fact_checkers_step1_vs_step2,
        "fact_checkers_step1_vs_step3": fact_checkers_step1_vs_step3,
        "fact_checkers_step2_vs_step3": fact_checkers_step2_vs_step3,
    }


def compute_single_pool_confusion_matrix(
    summary_dataset: dict[ItemT, dict[AnnotationT, int]],
    *,
    labels: list[AnnotationT],
    kind: Literal["micro", "macro"],
) -> dict[AnnotationT, dict[AnnotationT, int | float]]:
    """Compute confusion matrix for single-pool summary dataset.

    This is a pseudo- or modified confusion matrix, i.e. one where everything below the diagonal is
    zero, because we don't arbitrarily distinguish annotators from the same pool.
    """
    assert set(labels).issuperset({label for js in summary_dataset.values() for label in js})

    result: dict[AnnotationT, dict[AnnotationT, int | float]] = _empty_confusion_matrix(
        labels, labels
    )
    for _item, judgments in summary_dataset.items():
        total_item_pairs = math.comb(sum(judgments.values()), 2)
        for i, label1 in enumerate(labels):
            for label2 in labels[i:]:
                pairs_count = (
                    judgments.get(label1, 0) * judgments.get(label2, 0)
                    if label1 != label2
                    else math.comb(judgments.get(label1, 0), 2)
                )
                if kind == "macro" and total_item_pairs:
                    pairs_count /= total_item_pairs
                row = result.setdefault(label1, {})
                row[label2] = row.get(label2, 0) + pairs_count
                if label1 != label2:
                    other_row = result.setdefault(label2, {})
                    other_row[label1] = other_row.get(label1, 0) + pairs_count

    return result


def compute_cross_pool_confusion_matrix(
    summary_dataset1: dict[ItemT, dict[AnnotationT, int]],
    summary_dataset2: dict[ItemT, dict[AnnotationT, int]],
    *,
    labels: list[AnnotationT],
    kind: Literal["micro", "macro"],
) -> dict[AnnotationT, dict[AnnotationT, int | float]]:
    """Compute confusion matrix for cross-pool summary datasets."""
    assert set(labels).issuperset({label for js in summary_dataset1.values() for label in js})
    assert set(labels).issuperset({label for js in summary_dataset2.values() for label in js})

    items = sorted(summary_dataset1.keys() & summary_dataset2.keys())
    intersected_dataset_x = {item: summary_dataset1[item] for item in items}
    intersected_dataset_y = {item: summary_dataset2[item] for item in items}

    result: dict[AnnotationT, dict[AnnotationT, int | float]] = _empty_confusion_matrix(
        labels, labels
    )
    for item, judgments_x in intersected_dataset_x.items():
        judgments_y = intersected_dataset_y[item]
        total_item_pairs = sum(judgments_x.values()) * sum(judgments_y.values())
        for label_x in labels:
            for label_y in labels:
                pairs_count = judgments_x.get(label_x, 0) * judgments_y.get(label_y, 0)
                if kind == "macro" and total_item_pairs:
                    pairs_count /= total_item_pairs
                row = result.setdefault(label_x, {})
                row[label_y] = row.get(label_y, 0) + pairs_count

    return result


def agreement_confusion_matrices(
    annotation_data: list[EnhancedResponse],
    medical_expert_data: list[list[EnhancedResponse]],
    factchecking_data: list[list[EnhancedResponse]],
) -> dict[str, dict[bool, dict[bool, int | float]]]:
    """Calculate medical expert and factchecker confusion matrices."""
    sections = [
        section
        for response in annotation_data
        for section in response.annotations.sections_with_annotations
    ]
    first_pass_dataset = _first_pass_summary_dataset(sections)
    laj_dataset = _laj_summary_dataset(sections)
    medical_expert_dataset = _medical_expert_summary_dataset(medical_expert_data)
    factchecker_dataset = _factchecker_summary_dataset(factchecking_data)

    labels = [True, False]
    return {
        "first_pass_annotators_micro": compute_single_pool_confusion_matrix(
            first_pass_dataset, labels=labels, kind="micro"
        ),
        "first_pass_annotators_macro": compute_single_pool_confusion_matrix(
            first_pass_dataset, labels=labels, kind="macro"
        ),
        "medical_experts": compute_single_pool_confusion_matrix(
            medical_expert_dataset, labels=labels, kind="macro"
        ),
        "fact_checkers": compute_single_pool_confusion_matrix(
            factchecker_dataset, labels=labels, kind="macro"
        ),
        "first_pass_annotators_vs_medical_experts": compute_cross_pool_confusion_matrix(
            first_pass_dataset, medical_expert_dataset, labels=labels, kind="macro"
        ),
        "first_pass_annotators_vs_fact_checkers": compute_cross_pool_confusion_matrix(
            first_pass_dataset, factchecker_dataset, labels=labels, kind="macro"
        ),
        "laj_vs_medical_experts": compute_cross_pool_confusion_matrix(
            laj_dataset, medical_expert_dataset, labels=labels, kind="macro"
        ),
        "laj_vs_fact_checkers": compute_cross_pool_confusion_matrix(
            laj_dataset, factchecker_dataset, labels=labels, kind="macro"
        ),
        "medical_experts_vs_fact_checkers": compute_cross_pool_confusion_matrix(
            medical_expert_dataset, factchecker_dataset, labels=labels, kind="macro"
        ),
    }


def print_confusion_matrix(
    confusion_matrix: dict[Any, dict[Any, int | float | None]], *, side_label: str, top_label: str
) -> None:
    """Print confusion matrix as markdown."""
    rows = confusion_matrix.keys()
    columns = []
    for row in confusion_matrix.values():
        columns.extend([key for key in row if key not in columns])

    # Assemble and print header
    line = [rf"| {side_label} \/ vs. {top_label} -> |"]
    separator_line = ["|:--- |"]
    for column in columns:
        line.append(f" {column} |")
        separator_line.append(":--- |")
    print("".join(line))
    print("".join(separator_line))

    for row in rows:
        line = [f"| {row} |"]
        for column in columns:
            value = confusion_matrix[row].get(column, None)
            line.append(f" {value} |")
        print("".join(line))


def format_statistic_value(value: Any) -> str:
    """Format scalar and interval statistic values for console output."""
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, tuple) and len(value) == 2:
        if all(isinstance(item, int) for item in value):
            return str(value)
        try:
            lower, upper = value
            return f"({float(lower):.3f}, {float(upper):.3f})"
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def print_markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Print dictionaries as a Markdown table with the requested column order."""
    print(f"| {' | '.join(columns)} |")
    print(f"| {' | '.join(':---' for _ in columns)} |")
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, None)
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        print(f"| {' | '.join(values)} |")


def collate_adjudications(
    medical_expert_data: list[list[EnhancedResponse]],
) -> dict[str, list[SectionAdjudicationModel]]:
    """Collate adjudication data in order by time stamp."""
    result = {}
    for responses in medical_expert_data:
        for response in responses:
            for section in response.annotations.sections_with_annotations:
                if section.adjudication is not None:
                    result.setdefault(section.id, []).append(section.adjudication)
    for k, v in result.items():
        result[k] = sorted(v, key=lambda adj: adj.timestamp)
    return result


def collate_fact_checkings(
    factchecking_data: list[list[EnhancedResponse]],
) -> dict[str, list[SectionFactCheckingModel]]:
    """Collate fact-checking data in order by time stamp."""
    result = {}
    for responses in factchecking_data:
        for response in responses:
            for section in response.annotations.sections_with_annotations:
                if section.fact_checking is not None:
                    result.setdefault(section.id, []).append(section.fact_checking)
    for k, v in result.items():
        result[k] = sorted(v, key=lambda fc: fc.timestamp)
    return result


def plot_avg_response_accuracy_vs_section_accuracy(
    annotation_data: list[EnhancedResponse],
    medical_expert_data: list[list[EnhancedResponse]],
    factchecking_data: list[list[EnhancedResponse]],
) -> tuple[list[str], list[plt.Figure], list[plt.Axes]]:
    """Plot average of response accuracy annotations vs. % of sections labeled "hallucination".

    Plot six figures:

    1. Using medical experts' aggregate_judgment for long responses
    2. Using medical experts' aggregate_judgment for short responses
    3. Using fact-checkers' aggregate_judgment for long responses
    4. Using fact-checkers' aggregate_judgment for short responses
    5. Using both medical experts' and fact-checkers' aggregate_judgment for long responses
    6. Using both medical experts' and fact-checkers' aggregate_judgment for short responses

    We display these as heatmaps. We plot average scalar accuracy on the x axis. We plot
    percentage of sections judged incorrect on y axis. The color shows the number of responses in
    each bin.

    Long responses have at least three sections. Short responses have fewer than three sections.

    In each figure, we have one (x, y) pair per response in the data that has scoreable sections.
    If a section doesn't have fact-checking or adjudication information, but it has a consensus
    annotator label, we count that. We ignore unadjudicated sections without first-pass annotator
    consensus and we ignore sections marked not_claim by any human annotator. If a response has
    no scoreable sections, we ignore it.

    When we have multiple aggregate judgments of the same section, we count some sections as
    fractionally correct. We treat a label of "hallucination" as 1, a label of "no_hallucination"
    as 0, and we take the average as the section's label. This means our count of the "number of
    sections labeled hallucination" for that response used to calculate the percentage may be
    fractional.
    """
    names = [
        "medical_experts_long_responses",
        "medical_experts_short_responses",
        "fact_checkers_long_responses",
        "fact_checkers_short_responses",
        "both_long_responses",
        "both_short_responses",
    ]
    glosses = [
        "Average scalar accuracy vs. % of sections judged hallucination,\nlong responses, resolving disagreements with medical experts",
        "Average scalar accuracy vs. % of sections judged hallucination,\nshort responses, resolving disagreements with medical experts",
        "Average scalar accuracy vs. % of sections judged hallucination,\nlong responses, resolving disagreements with fact-checkers",
        "Average scalar accuracy vs. % of sections judged hallucination,\nshort responses, resolving disagreements with fact-checkers",
        "Average scalar accuracy vs. % of sections judged hallucination,\nlong responses, resolving disagreements with\nboth medical experts and fact-checkers",
        "Average scalar accuracy vs. % of sections judged hallucination,\nshort responses, resolving disagreements with\nboth medical experts and fact-checkers",
    ]
    # Only plot a callout box for the short-response plots
    do_callout_configs = [False, True, False, True, False, True]
    plot_configs = [
        (True, False, True),
        (True, False, False),
        (False, True, True),
        (False, True, False),
        (True, True, True),
        (True, True, False),
    ]
    judgment_mapping = {
        AdjudicationJudgment.HALLUCINATION: 1.0,
        AdjudicationJudgment.NO_HALLUCINATION: 0.0,
    }
    medical_expert_judgments: dict[str, list[float]] = {
        section_id: [judgment_mapping[judgment.aggregate_judgment] for judgment in judgments]
        for section_id, judgments in collate_adjudications(medical_expert_data).items()
    }
    factchecker_judgments: dict[str, list[float]] = {
        section_id: [judgment_mapping[judgment.aggregate_judgment] for judgment in judgments]
        for section_id, judgments in collate_fact_checkings(factchecking_data).items()
    }

    plot_data: list[list[tuple[float, float]]] = [[], [], [], [], [], []]
    n_section_labels_by_plot = [0, 0, 0, 0, 0, 0]
    for response in annotation_data:
        response_accuracies = []
        for annotation in response.annotations.answer_annotations:
            if (
                is_human(annotation)
                and annotation.dimensions is not None
                and annotation.dimensions.accuracy is not None
            ):
                response_accuracies.append(annotation.dimensions.accuracy)
        assert response_accuracies
        if len(response_accuracies) < 2:
            continue
        if (
            len(
                [
                    a
                    for a in response.annotations.sections_with_annotations[0].annotations
                    if is_human(a)
                ]
            )
            < 2
        ):
            continue
        avg_response_accuracy = float(np.mean(response_accuracies))
        is_long_response = (
            len(
                [
                    section
                    for section in response.annotations.sections_with_annotations
                    if all(
                        annotation.accuracy_type != IncorrectAnnotationType.NO_FACT
                        for annotation in section.annotations
                        if is_human(annotation)
                    )
                ]
            )
            >= 3
        )

        for plot_index, (use_medical_experts, use_factcheckers, plot_long_responses) in enumerate(
            plot_configs
        ):
            if is_long_response != plot_long_responses:
                continue
            section_labels = []
            for section in response.annotations.sections_with_annotations:
                if (
                    all(
                        annotation.accuracy_type != IncorrectAnnotationType.NO_FACT
                        for annotation in section.annotations
                        if is_human(annotation)
                    )
                    and len([a for a in section.annotations if is_human(a)]) >= 2
                ):
                    judgments: list[float] = []
                    if use_medical_experts:
                        judgments.extend(medical_expert_judgments.get(section.id, []))
                    if use_factcheckers:
                        judgments.extend(factchecker_judgments.get(section.id, []))
                    labels = sorted(
                        {
                            says_has_error(annotation)
                            for annotation in section.annotations
                            if is_human(annotation)
                        }
                    )
                    if not judgments and len(labels) == 1:
                        judgments.append(float(labels[0]))
                    if judgments:
                        section_labels.append(float(np.mean(judgments)))

            if section_labels:
                plot_data[plot_index].append(
                    (avg_response_accuracy, 100 * float(np.mean(section_labels)))
                )
                n_section_labels_by_plot[plot_index] += len(section_labels)

    print(f"Number of responses by plot: {[len(plot_datum) for plot_datum in plot_data]}")
    print(f"Number of section labels by plot: {n_section_labels_by_plot}")

    figs = []
    axs = []
    for gloss, do_callout, points in zip(glosses, do_callout_configs, plot_data, strict=True):
        fig = plt.figure(layout="constrained")
        grid = fig.add_gridspec(2, 2, width_ratios=(1.5, 4), height_ratios=(1, 4), wspace=0)
        pct_hist_ax = fig.add_subplot(grid[1, 0])
        scalar_hist_ax = fig.add_subplot(grid[0, 1])
        ax = fig.add_subplot(grid[1, 1], sharex=scalar_hist_ax, sharey=pct_hist_ax)

        fig.suptitle(gloss)
        figs.append(fig)
        axs.append(ax)
        if points:
            called_out_point = (3.0, 0.0)
            # Sieve out the called-out point so it doesn't affect the color of other points.
            # Keeping it makes the color map difficult to read.
            xs = [
                resp_acc
                for resp_acc, sec_acc in points
                if not do_callout or (resp_acc, sec_acc) != called_out_point
            ]
            ys = [
                sec_acc
                for resp_acc, sec_acc in points
                if not do_callout or (resp_acc, sec_acc) != called_out_point
            ]
            n_bins = 10
            heatmap, x_edges, y_edges = np.histogram2d(
                xs,
                ys,
                bins=n_bins,
                range=((0.9, 3.1), (-5, 105)),
            )
            # We start with a default value that will probably work then set the values using an algorithm that
            # definitely will work.
            callout_x_edges = x_edges[-2:]
            callout_y_edges = y_edges[0:]
            if callout_x_edges[0] > called_out_point[0] or called_out_point[0] > callout_x_edges[1]:
                for start, end in pairwise(x_edges):
                    if start <= called_out_point[0] <= end:
                        callout_x_edges = [start, end]
                        break
            if callout_y_edges[0] > called_out_point[1] or called_out_point[1] > callout_y_edges[1]:
                for start, end in pairwise(y_edges):
                    if start <= called_out_point[1] <= end:
                        callout_y_edges = [start, end]
                        break

            n_in_callout_bin = len(
                [
                    point
                    for point in points
                    if callout_x_edges[0] <= point[0] <= callout_x_edges[1]
                    and callout_y_edges[0] <= point[1] <= callout_y_edges[1]
                ]
            )

            masked_heatmap = np.ma.masked_where(heatmap == 0, heatmap)
            boundaries = list(range(int(np.max(heatmap)) + 1))
            cmap = plt.colormaps["viridis"].resampled(len(boundaries) - 1)
            norm = BoundaryNorm(boundaries, cmap.N, clip=True)
            hist_xs = [resp_acc for resp_acc, sec_acc in points]
            hist_ys = [sec_acc for resp_acc, sec_acc in points]
            pct_hist_ax.hist(
                hist_ys, bins=y_edges, color="0.4", edgecolor="white", orientation="horizontal"
            )
            pct_hist_ax.invert_xaxis()
            pct_hist_ax.set_ylabel("% of sections labeled has_error")
            pct_hist_ax.set_xlabel("# Resp")
            pct_hist_ax.tick_params(axis="y", labelbottom=False)
            scalar_hist_ax.hist(hist_xs, bins=x_edges, color="0.4", edgecolor="white")
            scalar_hist_ax.set_ylabel("# Resp")
            scalar_hist_ax.tick_params(axis="x", labelbottom=False)
            mesh = ax.pcolormesh(
                x_edges, y_edges, masked_heatmap.T, cmap=cmap, norm=norm, shading="auto"
            )
            colorbar = fig.colorbar(mesh, ax=ax)
            colorbar.set_label("# Resp")
            if do_callout:
                # Separately plot the called-out point in a special color
                callout_patch_width = x_edges[1] - x_edges[0]
                callout_patch_height = y_edges[1] - y_edges[0]
                ax.add_patch(
                    patches.Rectangle(
                        (callout_x_edges[0], callout_y_edges[0]),
                        callout_patch_width,
                        callout_patch_height,
                        facecolor="r",
                        edgecolor="none",
                    )
                )
                ax.text(
                    callout_x_edges[0] + callout_patch_width / 2,
                    callout_y_edges[0] + callout_patch_height / 2,
                    str(n_in_callout_bin),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="k",
                    fontweight="bold",
                )
        # ax.set_ylabel("% of sections labeled has_error")
        ax.set_xlabel("Average scalar accuracy")
        ax.set_xlim(0.9, 3.1)
        ax.set_ylim(-5, 105)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.grid(alpha=0.25)

    response_accuracy_judgments_for_correlation = [x for x, _ in plot_data[4] + plot_data[5]]
    section_hallucination_judgments_for_correlation = [y for _, y in plot_data[4] + plot_data[5]]
    correlation = float(
        np.corrcoef(
            response_accuracy_judgments_for_correlation,
            section_hallucination_judgments_for_correlation,
        )[0, 1]
    )
    print(
        f"Correlation (long + short responses, both medical experts and fact-checkers): "
        f"{correlation:.3f}"
    )

    return names, figs, axs


def calculate_table2(section_level_stastistics_) -> tuple[list[dict[str, Any]], list[str]]:
    """Format statistics for printing Table 2."""
    columns = ["category", "count"]
    rows = [
        {
            "category": "# FP annotator disagreement",
            "count": section_level_stastistics_["n_claimful_sections_where_humans_disagree"],
        },
        {
            "category": "# Found only by LLM-as-a-judge",
            "count": section_level_stastistics_["n_laj_only_has_error"],
        },
        {
            "category": "# Augment Injected, missed by all",
            "count": section_level_stastistics_["n_injector_only_has_error"],
        },
        {
            "category": "# Found by FP (but not LaJ)",
            "count": section_level_stastistics_[
                "n_humans_say_error_while_injector_and_detector_say_no"
            ],
        },
    ]
    total_adjudicated = sum(row["count"] for row in rows)
    rows.append({"category": "# Total Sections for Adjudication", "count": total_adjudicated})
    return rows, columns


def counts_by_has_error_and_comment_category(
    data: list[list[EnhancedResponse]],
    *,
    answer_key_name: str,
    level: str = "section",
    filter_by_laj_decision: IncorrectAnnotationType | None = None,
) -> tuple[int, list[dict[str, Any]], list[str]]:
    """Compute counts by has_error label and comment category."""
    if answer_key_name not in {"adjudication", "fact_checking"}:
        raise ValueError(f"Unrecognized answer key name {answer_key_name!r}")
    if level != "section":
        raise ValueError(f"Unrecognized level {level!r}")

    id_to_index_to_type: dict[int, dict[str, Literal["has_error", "no_error"]]] = {}
    id_to_index_to_categories: dict[int, dict[str, set[AdjudicationCommentCategory]]] = {}
    for idx, responses in enumerate(data):
        for response in responses:
            for section in response.annotations.sections_with_annotations:
                if filter_by_laj_decision:
                    laj_decision = None
                    for annotation in section.annotations:
                        if annotation.annotator_id == LAJ_ID:
                            laj_decision = annotation.accuracy_type
                    if laj_decision is None:
                        raise ValueError(f"No LaJ decision for section {section.id}")
                    if laj_decision != filter_by_laj_decision:
                        continue

                index_to_type = id_to_index_to_type.setdefault(section.id, {})
                index_to_categories = id_to_index_to_categories.setdefault(section.id, {})
                assert idx not in index_to_type
                assert idx not in index_to_categories
                adjudication = getattr(section, answer_key_name, None)
                if adjudication:
                    aggregate_judgment = adjudication.aggregate_judgment
                    has_error: Literal["has_error", "no_error"] = (
                        "has_error"
                        if aggregate_judgment == AdjudicationJudgment.HALLUCINATION
                        else "no_error"
                    )
                    index_to_type[idx] = has_error

                    if answer_key_name == "adjudication":
                        categories = {
                            v
                            for vs in section.adjudication.q2_category_response.values()
                            for v in vs
                        }
                    elif answer_key_name == "fact_checking":
                        categories = {
                            v.simple_category for v in section.fact_checking.q2_category_response
                        }
                    else:
                        raise ValueError(f"Unrecognized answer key name {answer_key_name}")
                    index_to_categories[idx] = categories

    counts: Counter[tuple[str, AdjudicationCommentCategory | Literal["Total Error Judgments"]]] = (
        Counter()
    )
    total_sections_counted = 0
    for section_id, index_to_type in id_to_index_to_type.items():
        # If all adjudicators agree on the label of has_error or no_error:
        labels: list[Literal["has_error", "no_error"]] = list(set(index_to_type.values()))
        all_categories = set()
        for cs in id_to_index_to_categories[section_id].values():
            all_categories.update(cs)
        total_sections_counted += 1 if len(all_categories) > 0 else 0
        label = labels[0] if len(labels) == 1 else "disagree"
        counts.update((label, category) for category in sorted(all_categories))
        counts[label, "Total Error Judgments"] += 1 if all_categories else 0

    row_labels = ["has_error", "no_error", "disagree"]
    columns = [
        "Error Judgment",
        "Citations",
        "Other Medical Info",
        "Numeric Info",
        "Missing Info",
        "Clarity",
        "Other",
        "Total Error Judgments",
    ]
    rows = []
    for row_label in row_labels:
        rows.append(
            {
                "Error Judgment": row_label,
                **{
                    column: counts[row_label, AdjudicationCommentCategory(column)]
                    for column in columns
                    if column not in {"Error Judgment", "Total Error Judgments"}
                },
                "Total Error Judgments": counts[row_label, "Total Error Judgments"],
            }
        )

    return total_sections_counted, rows, columns


def comment_category_counts_by_reviewer_type(
    adjudication_data: list[list[EnhancedResponse]],
) -> tuple[int, dict[str, int], dict[str, Counter[AdjudicationCommentCategory]]]:
    """Count canonical medical-expert comment categories by reviewer type.

    A comment's canonical category set is the union of the categories assigned to
    it in every medical-expert adjudication. Comments from the hallucination
    injector are not one of the reviewer groups reported in the paper and are
    excluded.
    """
    comment_id_to_reviewer_type: dict[str, str] = {}
    injector_comment_ids: set[str] = set()
    unrecognized_reviewers = set()
    for responses in adjudication_data:
        for response in responses:
            if response.annotations is None:
                continue
            for section in response.annotations.sections_with_annotations:
                for annotation in section.annotations:
                    if annotation.annotator_id == INJECTOR_ID:
                        injector_comment_ids.add(annotation.id)
                        continue
                    reviewer_type = (
                        "LaJ"
                        if annotation.annotator_id == LAJ_ID
                        else first_pass_annotator_type(annotation)
                    )
                    if reviewer_type is None:
                        unrecognized_reviewers.add(annotation.annotator_id)
                        continue
                    previous_type = comment_id_to_reviewer_type.setdefault(
                        annotation.id, reviewer_type
                    )
                    if previous_type != reviewer_type:
                        raise ValueError(
                            f"Comment {annotation.id!r} has inconsistent reviewer types: "
                            f"{previous_type!r} and {reviewer_type!r}"
                        )
    print(f"Unrecognized reviewers: {unrecognized_reviewers}")

    comment_id_to_categories: dict[str, set[AdjudicationCommentCategory]] = {}
    for responses in adjudication_data:
        for response in responses:
            if response.annotations is None:
                continue
            for section in response.annotations.sections_with_annotations:
                if section.adjudication is None:
                    continue
                for comment_id, categories in section.adjudication.q2_category_response.items():
                    if comment_id not in comment_id_to_reviewer_type:
                        if comment_id in injector_comment_ids:
                            continue
                        raise ValueError(
                            f"Could not determine reviewer type for comment {comment_id!r}"
                        )
                    comment_id_to_categories.setdefault(comment_id, set()).update(categories)

    n_comments_by_reviewer_type = dict.fromkeys(COMMENT_REVIEWER_TYPE_ORDER, 0)
    category_counts_by_reviewer_type: dict[str, Counter[AdjudicationCommentCategory]] = {
        reviewer_type: Counter() for reviewer_type in COMMENT_REVIEWER_TYPE_ORDER
    }
    for comment_id, categories in comment_id_to_categories.items():
        reviewer_type = comment_id_to_reviewer_type[comment_id]
        n_comments_by_reviewer_type[reviewer_type] += 1
        category_counts_by_reviewer_type[reviewer_type].update(categories)

    return (
        sum(n_comments_by_reviewer_type.values()),
        n_comments_by_reviewer_type,
        category_counts_by_reviewer_type,
    )


def latex_pct_comments_by_reviewer_type_and_comment_category(
    adjudication_data: list[list[EnhancedResponse]],
) -> str:
    """Build appendix table showing comment-category percentages by reviewer type."""
    (
        n_comments,
        n_comments_by_reviewer_type,
        category_counts_by_reviewer_type,
    ) = comment_category_counts_by_reviewer_type(adjudication_data)
    result_lines = [
        r"""        \toprule
        & \multicolumn{4}{c}{\shortstack{Medical experts\\%d comments}} \\
        \cmidrule(r){2-5}
\multicolumn{1}{r}{\shortstack{\textit{Reviewer}\\[-2pt] \textit{type}}}
& \shortstack{AI\\researchers}
& \shortstack{Students}
& \shortstack{FP\\medical\\experts}
& \shortstack{\lajshort} \\

\multicolumn{1}{r}{\textit{Comments} ($n$)}"""
        % n_comments
    ]
    counts_row = [
        n_comments_by_reviewer_type[reviewer_type] for reviewer_type in COMMENT_REVIEWER_TYPE_ORDER
    ]
    result_lines.append("            & " + " & ".join(str(count) for count in counts_row) + r" \\")
    result_lines.append(r"        \midrule")
    for category in AdjudicationCommentCategory:
        percentages = [
            (
                100.0 * category_counts_by_reviewer_type[reviewer_type][category] / count
                if count
                else 0.0
            )
            for reviewer_type, count in zip(COMMENT_REVIEWER_TYPE_ORDER, counts_row, strict=True)
        ]
        result_lines.append(
            "        "
            + " & ".join([category.value, *(f"{percentage:.0f}\\%" for percentage in percentages)])
            + " \\\\"
        )

    result_lines.append(r"        \bottomrule")
    return "\n".join(result_lines)


def latex_pct_sections_by_has_error_and_comment_category(
    me_data: list[list[EnhancedResponse]],
    fc_data: list[list[EnhancedResponse]],
) -> str:
    """Build appendix table showing percentage of sections by has_error and comment category."""
    me_n_sections, me_rows, me_columns = counts_by_has_error_and_comment_category(
        me_data, answer_key_name="adjudication", level="section"
    )
    fc_n_sections, fc_rows, fc_columns = counts_by_has_error_and_comment_category(
        fc_data, answer_key_name="fact_checking", level="section"
    )
    result_lines = [
        r"""        \toprule
        & \multicolumn{3}{c}{\shortstack{Medical experts\\%d sections}}
        & \multicolumn{3}{c}{\shortstack{Fact-checkers\\%d sections}} \\
        \cmidrule(lr){2-4}
        \cmidrule(l){5-7}
\multicolumn{1}{r}{\shortstack{\textit{Error}\\[-2pt] \textit{judgment}}}
& \shortstack{Has\\error}
& \shortstack{No\\error}
& \shortstack{Dis-\\agree}
& \shortstack{Has\\error}
& \shortstack{No\\error}
& \shortstack{Dis-\\agree} \\

\multicolumn{1}{r}{\textit{Sections} ($n$)}"""
        % (me_n_sections, fc_n_sections)
    ]

    me_rows_transposed = zip(
        *[
            [
                row[column]
                for column in me_columns
                if column not in {"Error Judgment", "Total Error Judgments"}
            ]
            for row in me_rows
        ],
        strict=True,
    )
    fc_rows_transposed = zip(
        *[
            [
                row[column]
                for column in fc_columns
                if column not in {"Error Judgment", "Total Error Judgments"}
            ]
            for row in fc_rows
        ],
        strict=True,
    )
    rows = [
        [*me_row, *fc_row]
        for me_row, fc_row in zip(me_rows_transposed, fc_rows_transposed, strict=True)
    ]
    counts_row = [row["Total Error Judgments"] for row in me_rows] + [
        row["Total Error Judgments"] for row in fc_rows
    ]
    result_lines.append("            & " + " & ".join(str(c) for c in counts_row) + r" \\")
    result_lines.append(r"        \midrule")
    row_labels = [
        column for column in fc_columns if column not in {"Error Judgment", "Total Error Judgments"}
    ]
    for label, row in zip(row_labels, rows, strict=False):
        transformed = [
            label,
            *[
                f"{100.0 * cast(float, v / count):.0f}\\%"
                for v, count in zip(row, counts_row, strict=True)
            ],
        ]
        result_lines.append("        " + " & ".join(str(v) for v in transformed) + " \\\\")

    result_lines.append(r"        \bottomrule")
    return "\n".join(result_lines)


def latex_pct_sections_by_has_error_and_comment_category_laj_pos_and_neg(
    medexpert_adjudications: list[list[EnhancedResponse]],
) -> str:
    """Build appendix table showing percentage of sections by has_error and comment category."""
    fp_n_sections, fp_rows, fp_columns = counts_by_has_error_and_comment_category(
        medexpert_adjudications,
        answer_key_name="adjudication",
        level="section",
        filter_by_laj_decision=IncorrectAnnotationType.FACTUALLY_INCORRECT,
    )
    fp_rows.pop()
    fn_n_sections, fn_rows, fn_columns = counts_by_has_error_and_comment_category(
        medexpert_adjudications,
        answer_key_name="adjudication",
        level="section",
        filter_by_laj_decision=IncorrectAnnotationType.FACTUALLY_CORRECT,
    )
    fn_rows.pop()

    result_lines = [
        r"""        \toprule
        & \multicolumn{2}{c}{\shortstack{\lajshort FPos\\%d sections}}
        & \multicolumn{2}{c}{\shortstack{\lajshort FNeg\\%d sections}} \\
        \cmidrule(lr){2-3}
        \cmidrule(lr){4-5}
\multicolumn{1}{r}{\shortstack{\textit{Error}\\[-2pt] \textit{judgment}}}
& \shortstack{Has\\error}
& \shortstack{No\\error}
& \shortstack{Has\\error}
& \shortstack{No\\error} \\

\multicolumn{1}{r}{\textit{Sections} ($n$)}"""
        % (fp_n_sections, fn_n_sections)
    ]

    fp_rows_transposed = zip(
        *[
            [
                row[column]
                for column in fp_columns
                if column not in {"Error Judgment", "Total Error Judgments"}
            ]
            for row in fp_rows
        ],
        strict=True,
    )
    fn_rows_transposed = zip(
        *[
            [
                row[column]
                for column in fn_columns
                if column not in {"Error Judgment", "Total Error Judgments"}
            ]
            for row in fn_rows
        ],
        strict=True,
    )
    rows = [
        [*fp_row, *fn_row]
        for fp_row, fn_row in zip(fp_rows_transposed, fn_rows_transposed, strict=True)
    ]
    counts_row = [row["Total Error Judgments"] for row in fp_rows] + [
        row["Total Error Judgments"] for row in fn_rows
    ]
    result_lines.append("            & " + " & ".join(str(c) for c in counts_row) + r" \\")
    result_lines.append(r"        \midrule")
    row_labels = [
        column for column in fp_columns if column not in {"Error Judgment", "Total Error Judgments"}
    ]
    for label, row in zip(row_labels, rows, strict=False):
        transformed = [
            label,
            *[
                f"{100.0 * cast(float, v / count):.0f}\\%"
                for v, count in zip(row, counts_row, strict=True)
            ],
        ]
        result_lines.append("        " + " & ".join(str(v) for v in transformed) + " \\\\")

    result_lines.append(r"        \bottomrule")
    return "\n".join(result_lines)


def main() -> None:
    """Script entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-file",
        type=Path,
        required=True,
        help="JSONL file of EnhancedResponse data.",
    )
    parser.add_argument(
        "--adjudication-files",
        type=Path,
        nargs="+",
        required=True,
        help="JSONL files of EnhancedResponse data with adjudication responses from medical experts.",
    )
    parser.add_argument(
        "--factchecking-files",
        type=Path,
        nargs="+",
        required=True,
        help="JSONL files of EnhancedResponse data with fact-checking responses.",
    )
    parser.add_argument(
        "--medexpert-adjudication-files",
        type=Path,
        nargs="*",
        required=True,
        help="JSONL files of EnhancedResponse data with adjudication responses from medical experts on the MedExpert dataset.",
    )
    parser.add_argument(
        "--save-figures-to",
        type=Path,
        required=True,
        help="Where to save figures.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help=(
            f"Number of bootstrap samples to use for {BOOTSTRAP_CONFIDENCE_LEVEL:.0%} percentile confidence intervals "
            "on agreement statistics. Also used for first-pass annotator P/R/F-1. Use 0 "
            "to skip confidence intervals."
        ),
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help="Random seed for bootstrap confidence intervals.",
    )
    args = parser.parse_args()
    annotation_file: Path = args.annotation_file
    adjudication_files: list[Path] = args.adjudication_files
    factchecking_files: list[Path] = args.factchecking_files
    medexpert_adjudication_files: list[Path] = args.medexpert_adjudication_files
    save_figures_to: Path = args.save_figures_to
    bootstrap_samples: int = args.bootstrap_samples
    bootstrap_seed: int = args.bootstrap_seed

    if not annotation_file.is_file():
        raise FileNotFoundError(
            f"Annotation file {str(annotation_file)!r} does not exist or is not a file."
        )
    if bootstrap_samples < 0:
        raise ValueError("--agreement-bootstrap-samples must be non-negative.")

    save_figures_to.mkdir(exist_ok=True, parents=True)
    if not save_figures_to.is_dir():
        raise NotADirectoryError(
            f"--save-figures-to path {str(save_figures_to)!r} could not be created or is not a directory."
        )

    unused_figures_root = save_figures_to / "unused"
    unused_figures_root.mkdir(exist_ok=True, parents=True)

    not_file = [
        adjudication_file
        for adjudication_file in adjudication_files
        if not adjudication_file.is_file()
    ]
    if not_file:
        raise FileNotFoundError(
            f"Adjudication files {str(not_file)!r} do not exist or are not files."
        )

    not_file = [
        factchecking_file
        for factchecking_file in factchecking_files
        if not factchecking_file.is_file()
    ]
    if not_file:
        raise FileNotFoundError(
            f"Factchecking files {str(not_file)!r} do not exist or are not files."
        )

    with open(annotation_file, encoding="utf-8") as annotations_in:
        data = [EnhancedResponse.model_validate_json(line) for line in annotations_in]

    adjudications = []
    for adjudication_file in adjudication_files:
        with open(adjudication_file, encoding="utf-8") as adjudications_in:
            adjudications.append(
                [EnhancedResponse.model_validate_json(line) for line in adjudications_in]
            )

    factcheckings = []
    for factchecking_file in factchecking_files:
        with open(factchecking_file, encoding="utf-8") as factcheckings_in:
            factcheckings.append(
                [EnhancedResponse.model_validate_json(line) for line in factcheckings_in]
            )

    medexpert_adjudications = []
    for medexpert_adjudication_file in medexpert_adjudication_files:
        with open(medexpert_adjudication_file, encoding="utf-8") as medexpert_adjudications_in:
            medexpert_adjudications.append(
                [EnhancedResponse.model_validate_json(line) for line in medexpert_adjudications_in]
            )

    print("## Table 1: Overview of First-Pass Annotation")
    response_level_statistics_ = response_level_statistics(data)
    section_level_statistics_ = section_level_statistics(data)
    print_markdown_table(
        columns=["Datum", "1 FP Ann", "2+ FP Anns"],
        rows=[
            {
                "Datum": "# Responses",
                "1 FP Ann": response_level_statistics_["n_responses_single_annotated_per_section"],
                "2+ FP Anns": response_level_statistics_[
                    "n_responses_multiple_annotated_per_section"
                ],
            },
            {
                "Datum": "# Sections with Claims",
                "1 FP Ann": section_level_statistics_["n_claimful_sections_single_annotated"],
                "2+ FP Anns": section_level_statistics_["n_claimful_sections_multiple_annotated"],
            },
            {
                "Datum": "Ann per Response",
                "1 FP Ann": 1,
                "2+ FP Anns": f"2-{response_level_statistics_['max_response_annotators']}",
            },
            {
                "Datum": "Resp. Agmnt. (α, ordinal)",
                "1 FP Ann": "-",
                "2+ FP Anns": response_level_statistics_["accuracy_agreement"],
            },
            {
                "Datum": "Sect. Agmnt. (α, nominal)",
                "1 FP Ann": "-",
                "2+ FP Anns": section_level_statistics_["human_ann_has_error_agreement"],
            },
        ],
    )
    print()

    print("### Detailed response-level statistics")
    for k, v in response_level_statistics_.items():
        print(f"{k}: {v}")
    print()

    print("### Detailed section-level statistics")
    for k, v in section_level_statistics_.items():
        print(f"{k}: {v}")
    print()

    print("## Table 2: Composition of the Adjudication Pool")
    table2_rows, table2_columns = calculate_table2(section_level_statistics_)
    print_markdown_table(rows=table2_rows, columns=table2_columns)
    print()

    print("## Tables 3 and 4")
    print("### Tables 3 and 4 upper halves: Agreement numbers")
    for name, matrix in agreement_confusion_matrices(data, adjudications, factcheckings).items():
        print(f"### {name} (# and %)")
        # Convert from confusion matrix to a table like what is in the paper
        row_labels: list[str]
        row_index_pairs: list[tuple[bool, bool]]
        round_to: int
        if "_vs_" in name:
            row_labels = ["Agree:Err=True", "Agree:Err=False", "Disagree:TF", "Disagree:FT"]
            row_index_pairs = [(True, True), (False, False), (True, False), (False, True)]
            assert len(row_labels) == len(row_index_pairs)
            round_to = 2
        else:
            row_labels = ["Agree:Err=True", "Agree:Err=False", "Disagree"]
            row_index_pairs = [(True, True), (False, False), (True, False)]
            assert len(row_labels) == len(row_index_pairs)
            round_to = 1
        numbers = [matrix[mrowidx][mcolidx] for mrowidx, mcolidx in row_index_pairs]
        percentages = [round(100 * number / sum(numbers), round_to) for number in numbers]
        rows = [
            {
                "Agreement?": row_label,
                "#": number,
                "%": percentage,
            }
            for row_label, number, percentage in zip(row_labels, numbers, percentages, strict=True)
        ]
        print_markdown_table(rows=rows, columns=["Agreement?", "#", "%"])
        print()
    print()

    print("### Tables 3 and 4 lower halves, plus Table 7: Agreement statistics")
    if bootstrap_samples:
        print(
            f"bootstrap_ci: {BOOTSTRAP_CONFIDENCE_LEVEL:.0%} percentile, "
            f"samples={bootstrap_samples}, seed={bootstrap_seed}"
        )
    else:
        print("bootstrap_ci: skipped")
    for k1, vs in agreement_statistics(
        data,
        adjudications,
        factcheckings,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    ).items():
        print(f"{k1}:")
        for k2, v in vs.items():
            formatted_v = format_statistic_value(v)
            print(f"  {k2}: {formatted_v}")
    print()

    print("## Figures 3, 4, and 5: Modified Confusion Matrices")
    matrix_labels = {
        "medical_experts_step1_vs_step2": {
            "side_label": "Step 1: Accuracy Error?",
            "top_label": "Step 2: Accuracy With Comments As Context",
        },
        "fact_checkers_step1_vs_step2": {
            "side_label": "Step 1: Accuracy Error?",
            "top_label": "Step 2: Accuracy With Respect to Excerpt",
        },
        "fact_checkers_step1_vs_step3": {
            "side_label": "Step 1: Accuracy Error?",
            "top_label": "Step 3: Accuracy With Respect To Comments",
        },
        "fact_checkers_step2_vs_step3": {
            "side_label": "Step 2: Accuracy With Respect To Excerpt",
            "top_label": "Step 3: Accuracy With Respect To Comments",
        },
    }
    for name, matrix in confusion_matrices(adjudications, factcheckings).items():
        print(f"### {name} (%)")
        total = sum(v for c_to_v in matrix.values() for v in c_to_v.values())
        percent_matrix = {
            r: {c: round(100 * v / total, 0) for c, v in c_to_v.items()}
            for r, c_to_v in matrix.items()
        }
        print_confusion_matrix(percent_matrix, **matrix_labels[name])
        print()
        print(f"### {name} (#)")
        print_confusion_matrix(matrix, **matrix_labels[name])
        print()
    print()

    print("## Appendix C.4: FC % marked error")
    fc_marked_has_error = [
        section.fact_checking.aggregate_judgment == AdjudicationJudgment.HALLUCINATION
        for factchecking in factcheckings
        for response in factchecking
        for section in response.annotations.sections_with_annotations
        if section.fact_checking is not None
    ]
    n_marked_error = sum(fc_marked_has_error)
    print(f"{n_marked_error / len(fc_marked_has_error):.0%} ({n_marked_error} / {len(fc_marked_has_error)})")
    print()

    print("## Figure 2: Average scalar accuracy vs. % of sections labeled hallucination")
    names, figs, _axs = plot_avg_response_accuracy_vs_section_accuracy(
        data, adjudications, factcheckings
    )
    for name, fig in zip(names, figs, strict=True):
        fig.savefig(
            unused_figures_root
            / f"avg_scalar_accuracy_vs_pct_labeled_hallucination_{name}_titled.pdf",
            bbox_inches="tight",
        )
        fig.savefig(
            unused_figures_root
            / f"avg_scalar_accuracy_vs_pct_labeled_hallucination_{name}_titled.png",
            bbox_inches="tight",
        )
        fig.suptitle("")
        fig.savefig(
            unused_figures_root / f"avg_scalar_accuracy_vs_pct_labeled_hallucination_{name}.pdf",
            bbox_inches="tight",
        )
    assert names[4] == "both_long_responses"
    assert names[5] == "both_short_responses"
    copy2(
        unused_figures_root / f"avg_scalar_accuracy_vs_pct_labeled_hallucination_{names[4]}.pdf",
        save_figures_to / f"avg_scalar_accuracy_vs_pct_labeled_hallucination_{names[4]}.pdf",
    )
    figs[5].suptitle("")
    w, h = figs[5].get_size_inches()
    figs[5].set_size_inches(w, h / 2)
    figs[5].savefig(
        save_figures_to / f"avg_scalar_accuracy_vs_pct_labeled_hallucination_{names[5]}.pdf",
        bbox_inches="tight",
    )
    print()

    print("## Appendix: Table 6: Response breakdown by augmentation and model")
    print_markdown_table(
        response_counts_by_model_and_augmentation(data),
        ["model", *RESPONSE_AUGMENTATION_ORDER, "total"],
    )
    print()

    print("## Appendix: Tables 9 and 11 showing errors by comment category")
    print("### Section-level counts: Medical experts")
    n_sections, rows, columns = counts_by_has_error_and_comment_category(
        adjudications, answer_key_name="adjudication", level="section"
    )
    print(f"Counted over {n_sections} sections with at least one adjudication")
    print_markdown_table(rows, columns)
    print()
    print("### Section-level counts: Fact-checking")
    n_sections, rows, columns = counts_by_has_error_and_comment_category(
        factcheckings, answer_key_name="fact_checking", level="section"
    )
    print(f"Counted over {n_sections} sections with at least one adjudication")
    print_markdown_table(rows, columns)

    print()
    print("### Section-level percentages: Table 9 as LaTeX")
    print("```latex")
    print(latex_pct_sections_by_has_error_and_comment_category(adjudications, factcheckings))
    print("```")
    print()
    print("### Comment-level percentages by reviewer type: Table 11 as LaTeX")
    print("```latex")
    print(latex_pct_comments_by_reviewer_type_and_comment_category(adjudications))
    print("```")
    if medexpert_adjudications:
        print()
        print("## Appendix: Extra MedExpert numbers")
        print("### Appendix: Tables showing errors by comment category")
        print("#### Section-level counts: Medical experts (LaJ false positives only)")
        n_sections, rows, columns = counts_by_has_error_and_comment_category(
            medexpert_adjudications,
            answer_key_name="adjudication",
            level="section",
            filter_by_laj_decision=IncorrectAnnotationType.FACTUALLY_INCORRECT,
        )
        print(f"Counted over {n_sections} adjudicated LaJ false-positive sections")
        print_markdown_table(rows, columns)
        print()
        print("#### Section-level counts: Medical experts (LaJ false negatives only)")
        n_sections, rows, columns = counts_by_has_error_and_comment_category(
            medexpert_adjudications,
            answer_key_name="adjudication",
            level="section",
            filter_by_laj_decision=IncorrectAnnotationType.FACTUALLY_CORRECT,
        )
        print(f"Counted over {n_sections} adjudicated LaJ false-negative sections")
        print_markdown_table(rows, columns)
        print()
        print("#### Section-level percentages: Table 10 as LaTeX")
        print("```latex")
        print(
            latex_pct_sections_by_has_error_and_comment_category_laj_pos_and_neg(
                medexpert_adjudications
            )
        )
        print("```")
    print()
    print("## Appendix: Table 13: First-pass annotator statistics")
    n_annotators, n_annotators_by_type = count_distinct_first_pass_annotators_by_type(data)
    print(f"n_distinct_first_pass_annotators: {n_annotators}")
    print("n_distinct_first_pass_annotators_by_type:")
    for annotator_type in ANNOTATOR_TYPE_ORDER:
        print(f"  {annotator_type}: {n_annotators_by_type[annotator_type]}")
    print()

    print("### Table 13 upper half: Distinct first-pass annotators by annotator type and level")
    print_markdown_table(
        distinct_first_pass_annotators_by_type_and_level(data),
        [
            "annotator_type",
            "response_level_annotators",
            "section_level_annotators",
            "any_level_annotators",
        ],
    )
    print()

    print("### Table 13 upper half: First-pass annotation coverage by annotator type")
    ai_researcher, students, medical_experts = first_pass_annotation_coverage_by_type(data)
    assert ai_researcher["annotator_type"] == "AI Researcher"
    assert students["annotator_type"] == "Student"
    assert medical_experts["annotator_type"] == "Medical Expert"
    measures_of_interest = [
        "distinct_annotators",
        "section_level_multiple_annotated_sections",
        "section_level_single_annotated_sections",
    ]
    measure_glosses = {
        "distinct_annotators": "Annotators, n",
        "section_level_multiple_annotated_sections": "Multiply Ann. Sec.",
        "section_level_single_annotated_sections": "Singly Ann. Sec.",
    }
    columns = ("Measure", "AI researchers", "Students", "FP medical experts")
    print(f"DEBUG: ai_researcher={ai_researcher}")
    rows = [
        {
            "Measure": measure_glosses[measure],
            "AI researchers": ai_researcher[measure],
            "Students": students[measure],
            "FP medical experts": medical_experts[measure],
        }
        for measure in measures_of_interest
    ]
    print_markdown_table(rows, columns)
    print()

    print("### Full details: First-pass annotation coverage by annotator type")
    print_markdown_table(
        first_pass_annotation_coverage_by_type(data),
        [
            "annotator_type",
            "distinct_annotators",
            "n_response_judgments",
            "n_section_judgments",
            "response_level_single_annotated_responses",
            "section_level_single_annotated_responses",
            "response_level_multiple_annotated_responses",
            "section_level_multiple_annotated_responses",
            "section_level_single_annotated_sections",
            "section_level_multiple_annotated_sections",
        ],
    )
    print()

    fp_prf1_by_annotator_type = first_pass_precision_recall_f1_by_type(
        data,
        adjudications,
        factcheckings,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    ci_desc = format(BOOTSTRAP_CONFIDENCE_LEVEL, ".0%")

    print("### Table 13 lower half: FP Ann precision/recall/F1 by annotator type")
    print_markdown_table(
        fp_prf1_by_annotator_type,
        [
            "answer_key",
            "annotator_type",
            "precision",
            f"precision_bootstrap{ci_desc}ci_responselevel",
            "recall",
            f"recall_bootstrap{ci_desc}ci_responselevel",
            "f1",
            f"f1_bootstrap{ci_desc}ci_responselevel",
        ],
    )
    print()

    print("### Full details: FP Ann precision/recall/F1 by annotator type")
    print_markdown_table(
        fp_prf1_by_annotator_type,
        [
            "answer_key",
            "annotator_type",
            "precision",
            f"precision_bootstrap{ci_desc}ci_responselevel",
            f"precision_bootstrap{ci_desc}ci_sectionlevel",
            "recall",
            f"recall_bootstrap{ci_desc}ci_responselevel",
            f"recall_bootstrap{ci_desc}ci_sectionlevel",
            "f1",
            f"f1_bootstrap{ci_desc}ci_responselevel",
            f"f1_bootstrap{ci_desc}ci_sectionlevel",
            "n_answer_key_sections",
            "n_sections_with_predictions",
            "n_prediction_rows",
        ],
    )


if __name__ == "__main__":
    main()
