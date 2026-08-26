"""Score first-pass annotators and a system against section-level answer keys.

Inputs follow the same split-file convention as `run_hallucinations_analyses.py`:

- `--annotation-file` is a JSONL file of `EnhancedResponse` data containing first-pass
  section annotations and the system annotation named by `--pred-annotator-id`.
  Note that `--pred-annotator-id` is required for scoring --- you can't score the first-pass
  annotators without specifying a system annotator ID to score.
- `--adjudication-files` are JSONL files of `EnhancedResponse` data containing medical-expert
  adjudication judgments.
- `--factchecking-files` are JSONL files of `EnhancedResponse` data containing fact-checker
  judgments.

The script writes `scores.csv` with precision, recall, F1, number of answer-key sections, and
number of prediction rows. It builds three binary section-level answer keys:

- `first_pass`: first-pass human annotator consensus.
- `first_pass_plus_medical_experts`: unanimous medical-expert aggregate judgments where present,
  otherwise first-pass human annotator consensus. Excludes sections with disagreement between
  adjudicators and excluding sections with un-adjudicated first-pass annotation disagreements.
- `first_pass_plus_fact_checkers`: unanimous fact-checker aggregate judgments where present,
  otherwise first-pass human annotator consensus. Excluding sections with disagreement between
  adjudicators and excluding sections with un-adjudicated first-pass annotation disagreements.

The Answer key includes:
- Anything with consensus label among first-pass annotators.
- Adjudicated judgments where there is agreement.

Sections are excluded from all answer keys when they have fewer than
`--min-num-human-annotations` first-pass human annotations or any first-pass human annotator
marks them as not containing a factual claim. Sections with non-unanimous medical-expert or
fact-checker judgments are excluded from the corresponding augmented answer key.
"""

import argparse
from collections.abc import Callable, Iterable, Sequence
import csv
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from mdhjudgments.model import (
    AdjudicationJudgment,
    EnhancedResponse,
    IncorrectAnnotationType,
    InformationAccuracyAnnotationModel,
    SectionAdjudicationModel,
    SectionFactCheckingModel,
    SectionResponseAnnotation,
)
from mdhjudgments.run_hallucinations_analyses import (
    BOOTSTRAP_CONFIDENCE_LEVEL,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    bootstrap_confidence_intervals,
)

LAJ_ID = "IsiHallucinationDetector"
INJECTOR_ID = "IsiHallucinationInjector"


class AnswerKeyName(StrEnum):
    """Names of answer keys emitted in the score file."""

    # Ignore "hardcoded password" warnings, because these
    # aren't related to passwords at all
    FIRST_PASS_ANNOTATORS = "first_pass_annotators"  # noqa: S105
    FIRST_PASS_ANNOTATORS_PLUS_MEDICAL_EXPERTS = (
        "first_pass_annotators_plus_medical_experts"  # noqa: S105
    )
    FIRST_PASS_ANNOTATORS_PLUS_FACT_CHECKERS = (
        "first_pass_annotators_plus_fact_checkers"  # noqa: S105
    )


@dataclass(frozen=True)
class ScoreItem:
    """A scoreable section for one answer key."""

    response_id: str
    section_id: str
    gold: bool
    human_predictions: list[bool]
    system_prediction: bool


def _read_enhanced_responses(file_name: Path) -> list[EnhancedResponse]:
    """Read an EnhancedResponse JSONL file."""
    with open(file_name, encoding="utf-8") as responses_in:
        return [EnhancedResponse.model_validate_json(line) for line in responses_in]


def _validate_files(paths: Sequence[Path], *, label: str) -> None:
    """Raise if any input path is not a file."""
    not_file = [path for path in paths if not path.is_file()]
    if not_file:
        raise FileNotFoundError(f"{label} {str(not_file)!r} do not exist or are not files.")


def _has_detector_in_ann_id(annotation: InformationAccuracyAnnotationModel) -> bool:
    return "detector" in annotation.annotator_id.lower()


def _is_human_annotation(
    annotation: InformationAccuracyAnnotationModel, *, pred_annotator_id: str
) -> bool:
    return annotation.annotator_id not in {
        LAJ_ID,
        INJECTOR_ID,
        pred_annotator_id,
    } and not _has_detector_in_ann_id(annotation)


def _get_human_annotations(
    section: SectionResponseAnnotation, *, pred_annotator_id: str
) -> list[InformationAccuracyAnnotationModel]:
    return [
        annotation
        for annotation in section.annotations
        if _is_human_annotation(annotation, pred_annotator_id=pred_annotator_id)
    ]


def _get_system_annotation(
    section: SectionResponseAnnotation, *, pred_annotator_id: str
) -> InformationAccuracyAnnotationModel:
    system_annotations = [
        annotation
        for annotation in section.annotations
        if annotation.annotator_id == pred_annotator_id
    ]
    if len(system_annotations) != 1:
        raise ValueError(
            f"Expected exactly one {pred_annotator_id!r} annotation for section "
            f"{section.id!r}, got {len(system_annotations)}."
        )
    return system_annotations[0]


def _annotation_indicates_section_has_problem(
    annotation: InformationAccuracyAnnotationModel,
) -> bool:
    return (
        annotation.accuracy_type == IncorrectAnnotationType.FACTUALLY_INCORRECT
        or not annotation.element_correctness.certainty
        or not annotation.element_correctness.risk
        or not annotation.element_correctness.urgency
    )


def _collate_adjudications(
    adjudication_data: Iterable[list[EnhancedResponse]],
) -> dict[str, list[SectionAdjudicationModel]]:
    """Collate adjudication data in order by time stamp."""
    result: dict[str, list[SectionAdjudicationModel]] = {}
    for responses in adjudication_data:
        for response in responses:
            if response.annotations is None:
                continue
            for section in response.annotations.sections_with_annotations:
                if section.adjudication is not None:
                    result.setdefault(section.id, []).append(section.adjudication)
    for section_id, adjudications in result.items():
        result[section_id] = sorted(adjudications, key=lambda adjudication: adjudication.timestamp)
    return result


def _collate_fact_checkings(
    factchecking_data: Iterable[list[EnhancedResponse]],
) -> dict[str, list[SectionFactCheckingModel]]:
    """Collate fact-checking data in order by time stamp."""
    result: dict[str, list[SectionFactCheckingModel]] = {}
    for responses in factchecking_data:
        for response in responses:
            if response.annotations is None:
                continue
            for section in response.annotations.sections_with_annotations:
                if section.fact_checking is not None:
                    result.setdefault(section.id, []).append(section.fact_checking)
    for section_id, fact_checkings in result.items():
        result[section_id] = sorted(
            fact_checkings, key=lambda fact_checking: fact_checking.timestamp
        )
    return result


def _consensus_bool(labels: Sequence[bool]) -> bool | None:
    """Return the unanimous label, or None when labels are empty or disagree."""
    if not labels:
        return None
    if len(set(labels)) != 1:
        return None
    return labels[0]


def _aggregate_judgment_consensus(
    judgments: Sequence[SectionAdjudicationModel | SectionFactCheckingModel],
) -> bool | None:
    """Return unanimous aggregate judgment as a boolean hallucination label."""
    return _consensus_bool(
        [
            judgment.aggregate_judgment == AdjudicationJudgment.HALLUCINATION
            for judgment in judgments
        ]
    )


def _gold_label(
    answer_key_name: AnswerKeyName,
    *,
    min_num_human_annotations: int,
    first_pass_annotations: Sequence[InformationAccuracyAnnotationModel],
    adjudications: Sequence[SectionAdjudicationModel],
    fact_checkings: Sequence[SectionFactCheckingModel],
) -> bool | None:
    """Return the gold label for a section under an answer-key configuration."""
    if len(first_pass_annotations) < min_num_human_annotations:
        return None

    if any(
        annotation.accuracy_type == IncorrectAnnotationType.NO_FACT
        for annotation in first_pass_annotations
    ):
        return None

    first_pass_consensus = _consensus_bool(
        [
            _annotation_indicates_section_has_problem(annotation)
            for annotation in first_pass_annotations
        ]
    )
    if answer_key_name == AnswerKeyName.FIRST_PASS_ANNOTATORS:
        return first_pass_consensus

    if answer_key_name == AnswerKeyName.FIRST_PASS_ANNOTATORS_PLUS_MEDICAL_EXPERTS:
        if adjudications:
            return _aggregate_judgment_consensus(adjudications)
        return first_pass_consensus

    if answer_key_name == AnswerKeyName.FIRST_PASS_ANNOTATORS_PLUS_FACT_CHECKERS:
        if fact_checkings:
            return _aggregate_judgment_consensus(fact_checkings)
        return first_pass_consensus

    raise ValueError(f"Unexpected answer key: {answer_key_name!r}")


def _build_score_items(
    responses: list[EnhancedResponse],
    *,
    answer_key_name: AnswerKeyName,
    adjudications_by_section: dict[str, list[SectionAdjudicationModel]],
    fact_checkings_by_section: dict[str, list[SectionFactCheckingModel]],
    min_num_human_annotations: int,
    pred_annotator_id: str,
) -> list[ScoreItem]:
    """Build scoreable sections for one answer-key configuration."""
    score_items = []
    for response in responses:
        if response.annotations is None:
            continue
        for section in response.annotations.sections_with_annotations:
            human_annotations = _get_human_annotations(
                section,
                pred_annotator_id=pred_annotator_id,
            )
            gold = _gold_label(
                answer_key_name,
                min_num_human_annotations=min_num_human_annotations,
                first_pass_annotations=human_annotations,
                adjudications=adjudications_by_section.get(section.id, []),
                fact_checkings=fact_checkings_by_section.get(section.id, []),
            )
            if gold is None:
                continue

            system_annotation = _get_system_annotation(section, pred_annotator_id=pred_annotator_id)
            score_items.append(
                ScoreItem(
                    response_id=response.id,
                    section_id=section.id,
                    gold=gold,
                    human_predictions=[
                        _annotation_indicates_section_has_problem(annotation)
                        for annotation in human_annotations
                    ],
                    system_prediction=_annotation_indicates_section_has_problem(system_annotation),
                )
            )
    return score_items


def _compute_prf1(golds: Sequence[bool], preds: Sequence[bool]) -> tuple[float | None, ...]:
    """Compute binary precision/recall/F1, returning None metrics for empty inputs."""
    if len(golds) != len(preds):
        raise ValueError("Gold and prediction lists must be the same length.")
    if not golds:
        return None, None, None
    precision, recall, f1, _ = precision_recall_fscore_support(
        golds, preds, average="binary", zero_division=0.0
    )
    return float(precision), float(recall), float(f1)


def _score_human_annotations(score_items: Sequence[ScoreItem]) -> tuple[float | None, ...]:
    """Score all first-pass human section annotations against the answer key."""
    golds = []
    preds = []
    for item in score_items:
        golds.extend([item.gold for _ in item.human_predictions])
        preds.extend(item.human_predictions)
    return _compute_prf1(golds, preds)


def _score_system_annotations(score_items: Sequence[ScoreItem]) -> tuple[float | None, ...]:
    """Score system section annotations against the answer key."""
    return _compute_prf1(
        [item.gold for item in score_items],
        [item.system_prediction for item in score_items],
    )


def _bootstrap_ci_for_scores(
    score_items: Sequence[ScoreItem],
    *,
    score_fn: Callable[[Sequence[ScoreItem]], tuple[float | None, ...]],
    sample_over: Literal["responses", "sections"],
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[tuple[float | None, float | None] | None, ...]:
    if bootstrap_samples == 0:
        return (None, None), (None, None), (None, None)
    elif bootstrap_samples < 0:
        raise ValueError("Can't take negative bootstrap samples.")

    bootstrap_items: list[str] = sorted(
        {item.response_id for item in score_items}
        if sample_over == "responses"
        else {item.section_id for item in score_items}
    )
    ids_to_prf1: dict[tuple[str, ...], tuple[float | None, ...]] = {}

    if sample_over == "responses":
        mapping = {}
        for item in score_items:
            mapping.setdefault(item.response_id, []).append(item)
    elif sample_over == "sections":
        mapping = {item.section_id: [item] for item in score_items}
    else:
        raise ValueError(f"Unrecognized sample_over value {sample_over!r}")

    def _get_relevant_items(item_ids: Sequence[str], *, mapping=mapping) -> Sequence[ScoreItem]:
        return [item for item_id in item_ids for item in mapping[item_id]]

    def _lookup_or_compute_prf1(ids: list[str]) -> tuple[float | None, ...]:
        key = tuple(ids)
        result = ids_to_prf1.get(key)
        if result is None:
            result = ids_to_prf1[key] = score_fn(_get_relevant_items(ids))
        return result

    def _get_precision(response_ids: list[str]) -> float | None:
        precision, _, _ = _lookup_or_compute_prf1(response_ids)
        return precision

    def _get_recall(response_ids: list[str]) -> float | None:
        _, recall, _ = _lookup_or_compute_prf1(response_ids)
        return recall

    def _get_f1(response_ids: list[str]) -> float | None:
        _, _, f1 = _lookup_or_compute_prf1(response_ids)
        return f1

    cis = bootstrap_confidence_intervals(
        bootstrap_items,
        bootstrap_samples=bootstrap_samples,
        rng=rng,
        statistics={
            "precision": _get_precision,
            "recall": _get_recall,
            "f1": _get_f1,
        },
    )
    return cis["precision_ci_95"], cis["recall_ci_95"], cis["f1_ci_95"]


def _compute_scores(
    responses: list[EnhancedResponse],
    adjudication_data: list[list[EnhancedResponse]],
    factchecking_data: list[list[EnhancedResponse]],
    *,
    min_num_human_annotations: int,
    pred_annotator_id: str,
    output_dir: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> None:
    """Compute all score rows and write scores.csv."""
    adjudications_by_section = _collate_adjudications(adjudication_data)
    fact_checkings_by_section = _collate_fact_checkings(factchecking_data)

    rows = []
    for answer_key_name in AnswerKeyName:
        score_items = _build_score_items(
            responses,
            answer_key_name=answer_key_name,
            adjudications_by_section=adjudications_by_section,
            fact_checkings_by_section=fact_checkings_by_section,
            min_num_human_annotations=min_num_human_annotations,
            pred_annotator_id=pred_annotator_id,
        )
        n_sections = len(score_items)

        human_precision, human_recall, human_f1 = _score_human_annotations(score_items)
        for sample_over in ["responses", "sections"]:
            human_precision_ci, human_recall_ci, human_f1_ci = _bootstrap_ci_for_scores(
                score_items,
                score_fn=_score_human_annotations,
                sample_over=sample_over,
                rng=np.random.default_rng(bootstrap_seed),
                bootstrap_samples=bootstrap_samples,
            )
            rows.append(
                [
                    answer_key_name,
                    "first_pass_annotators",
                    human_precision,
                    *(human_precision_ci or ("n/a", "n/a")),
                    human_recall,
                    *(human_recall_ci or ("n/a", "n/a")),
                    human_f1,
                    *(human_f1_ci or ("n/a", "n/a")),
                    n_sections,
                    sum(len(item.human_predictions) for item in score_items),
                    sample_over,
                    bootstrap_samples,
                    bootstrap_seed,
                ]
            )

        system_precision, system_recall, system_f1 = _score_system_annotations(score_items)
        for sample_over in ["responses", "sections"]:
            system_precision_ci, system_recall_ci, system_f1_ci = _bootstrap_ci_for_scores(
                score_items,
                score_fn=_score_system_annotations,
                sample_over=sample_over,
                rng=np.random.default_rng(bootstrap_seed),
                bootstrap_samples=bootstrap_samples,
            )
            rows.append(
                [
                    answer_key_name,
                    pred_annotator_id,
                    system_precision,
                    *(system_precision_ci or ("n/a", "n/a")),
                    system_recall,
                    *(system_recall_ci or ("n/a", "n/a")),
                    system_f1,
                    *(system_f1_ci or ("n/a", "n/a")),
                    n_sections,
                    n_sections,
                    sample_over,
                    bootstrap_samples,
                    bootstrap_seed,
                ]
            )

    ci_desc = format(BOOTSTRAP_CONFIDENCE_LEVEL, ".0%")
    with open(output_dir / "scores.csv", "w", newline="") as scores_out:
        writer = csv.writer(scores_out)
        writer.writerow(
            [
                "answer_key",
                "predictor",
                "precision",
                f"precision_bootstrap{ci_desc}ci_lower",
                f"precision_bootstrap{ci_desc}ci_upper",
                "recall",
                f"recall_bootstrap{ci_desc}ci_lower",
                f"recall_bootstrap{ci_desc}ci_upper",
                "f1",
                f"f1_bootstrap{ci_desc}ci_lower",
                f"f1_bootstrap{ci_desc}ci_upper",
                "n_answer_key_sections",
                "n_prediction_rows",
                "bootstrap_ci_sampledover",
                "bootstrap_ci_samples",
                "bootstrap_ci_seed",
            ]
        )
        writer.writerows(rows)


def main() -> None:
    """Script entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-file",
        type=Path,
        required=True,
        help="JSONL file of EnhancedResponse data with first-pass annotations.",
    )
    parser.add_argument(
        "--adjudication-files",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "JSONL files of EnhancedResponse data with adjudication responses from medical "
            "experts."
        ),
    )
    parser.add_argument(
        "--factchecking-files",
        type=Path,
        nargs="+",
        required=True,
        help="JSONL files of EnhancedResponse data with fact-checking responses.",
    )
    parser.add_argument(
        "--pred-annotator-id",
        required=True,
        help="Annotator ID for the system annotations to score.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where scores.csv will be written.",
    )
    parser.add_argument("--min-num-human-annotations", type=int, default=2)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help=(
            f"Number of bootstrap samples to use for {BOOTSTRAP_CONFIDENCE_LEVEL:.0%} percentile confidence intervals "
            "on P/R/F-1. Use 0 to skip confidence intervals."
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
    output_dir: Path = args.output_dir

    _validate_files([annotation_file], label="Annotation file")
    _validate_files(adjudication_files, label="Adjudication files")
    _validate_files(factchecking_files, label="Factchecking files")

    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise NotADirectoryError(
            f"--output-dir path {str(output_dir)!r} could not be created or is not a directory."
        )

    annotations = _read_enhanced_responses(annotation_file)
    adjudications = [
        _read_enhanced_responses(adjudication_file) for adjudication_file in adjudication_files
    ]
    factcheckings = [
        _read_enhanced_responses(factchecking_file) for factchecking_file in factchecking_files
    ]

    _compute_scores(
        annotations,
        adjudications,
        factcheckings,
        min_num_human_annotations=args.min_num_human_annotations,
        pred_annotator_id=args.pred_annotator_id,
        output_dir=output_dir,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )


if __name__ == "__main__":
    main()
