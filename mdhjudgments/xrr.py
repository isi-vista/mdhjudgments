"""Python port of the official Google xRR Java library.

This mirrors the behavior of:
- DistanceFunctions
- XrrMetric / XrrMetrics
- XrrProcessor

The official repository is Java-only and archived; this module is a direct
Python translation of the published Java implementation.

Based on https://github.com/google-research/cross-replication-reliability.
Co-authored by GPT-5.5 (Instant).
"""

from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, MutableMapping
from enum import Enum
from typing import TypeVar

ItemT = TypeVar("ItemT", bound=Hashable)
AnnotationT = TypeVar("AnnotationT", bound=Hashable)
type SummaryDataset[ItemT, AnnotationT] = Mapping[ItemT, Mapping[AnnotationT, int]]
T = TypeVar("T")
type DistanceFunction[T] = Callable[[T, T], float]


class DistanceFunctions:
    """Common distance functions."""

    @staticmethod
    def interval_squared(a: T, b: T) -> float:
        """Squared interval distance function.

        Interprets the values as numbers and return their squared Euclidean distance.
        """
        return (float(a) - float(b)) ** 2

    @staticmethod
    def nominal(a: T, b: T) -> float:
        """Nominal distance function.

        This is the Kronecker delta, or the discrete distance function.
        """
        return 0.0 if a == b else 1.0


def _count_annotations(annotation_count_dict: Mapping[AnnotationT, int] | None) -> int:
    if annotation_count_dict is None:
        return 0
    return sum(int(x) for x in annotation_count_dict.values())


def _total_num_annotations(dataset: Mapping[ItemT, Mapping[AnnotationT, int]]) -> int:
    return sum(_count_annotations(v) for v in dataset.values())


def _marginal_expected_disagreement(
    distance_function: DistanceFunction[AnnotationT],
    annotation_nums_x: Mapping[AnnotationT, int],
    annotation_nums_y: Mapping[AnnotationT, int],
) -> float:
    sum_distance = 0.0
    for annotation_x, num_annotations_x in annotation_nums_x.items():
        for annotation_y, num_annotations_y in annotation_nums_y.items():
            distance = distance_function(annotation_x, annotation_y)
            sum_distance += distance * num_annotations_x * num_annotations_y
    return sum_distance


def _marginal_observed_disagreement(
    annotation_count_dict_x: Mapping[AnnotationT, int],
    annotation_count_dict_y: Mapping[AnnotationT, int],
    distance_function: DistanceFunction[AnnotationT],
) -> float:
    num_annotations_of_item_on_x = _count_annotations(annotation_count_dict_x)
    num_annotations_of_item_on_y = _count_annotations(annotation_count_dict_y)

    sum_distance = 0.0
    for annotation_x, annotation_count_x in annotation_count_dict_x.items():
        for annotation_y, annotation_count_y in annotation_count_dict_y.items():
            distance = distance_function(annotation_x, annotation_y)
            sum_distance += distance * annotation_count_x * annotation_count_y

    return (
        sum_distance
        * (num_annotations_of_item_on_x + num_annotations_of_item_on_y)
        / (num_annotations_of_item_on_x * num_annotations_of_item_on_y)
    )


class XrrMetrics(Enum):
    """Collection of Cross Replication Reliability metrics.

    The official Java library currently exposes only WITH_MISSING_DATA.
    """

    WITH_MISSING_DATA = "WITH_MISSING_DATA"

    def compute_xrr(
        self,
        dataset_x: SummaryDataset[ItemT, AnnotationT],
        dataset_y: SummaryDataset[ItemT, AnnotationT],
        distance_function: DistanceFunction[AnnotationT],
    ) -> float:
        """Compute xRR using this metric."""
        if self is not XrrMetrics.WITH_MISSING_DATA:
            raise NotImplementedError(f"Unsupported metric: {self}")

        intersection_items: set[ItemT] = dataset_x.keys() & dataset_y.keys()

        intersected_dataset_x = _get_intersected_dataset(dataset_x, intersection_items)
        intersected_dataset_y = _get_intersected_dataset(dataset_y, intersection_items)

        total_num_annotations_x = _total_num_annotations(intersected_dataset_x)
        total_num_annotations_y = _total_num_annotations(intersected_dataset_y)

        observed_disagreement = _compute_observed_disagreement(
            intersected_dataset_x,
            intersected_dataset_y,
            distance_function,
            total_num_annotations_x,
            total_num_annotations_y,
        )
        expected_disagreement = _compute_expected_disagreement(
            intersected_dataset_x,
            intersected_dataset_y,
            distance_function,
            total_num_annotations_x,
            total_num_annotations_y,
        )
        return float(1.0 - observed_disagreement / expected_disagreement)


def _get_intersected_dataset(
    dataset: SummaryDataset[ItemT, AnnotationT], intersection_items: set[ItemT]
) -> Mapping[ItemT, Mapping[AnnotationT, int]]:
    if len(dataset) == len(intersection_items):
        return dataset
    return {k: v for k, v in dataset.items() if k in intersection_items}


def _compute_expected_disagreement(
    intersected_dataset_x: SummaryDataset[ItemT, AnnotationT],
    intersected_dataset_y: SummaryDataset[ItemT, AnnotationT],
    distance_function: DistanceFunction[AnnotationT],
    total_num_annotations_x: int,
    total_num_annotations_y: int,
) -> float:
    return sum(
        sum(
            _marginal_expected_disagreement(distance_function, annotation_nums_x, annotation_nums_y)
            for annotation_nums_y in intersected_dataset_y.values()
        )
        for annotation_nums_x in intersected_dataset_x.values()
    ) / (total_num_annotations_x * total_num_annotations_y)


def _compute_observed_disagreement(
    intersected_dataset_x: SummaryDataset[ItemT, AnnotationT],
    intersected_dataset_y: SummaryDataset[ItemT, AnnotationT],
    distance_function: DistanceFunction[AnnotationT],
    total_num_annotations_x: int,
    total_num_annotations_y: int,
) -> float:
    sum_observed = 0.0
    for item in intersected_dataset_x:
        annotation_num_dict_x = intersected_dataset_x[item]
        annotation_num_dict_y = intersected_dataset_y[item]
        sum_observed += _marginal_observed_disagreement(
            annotation_num_dict_x, annotation_num_dict_y, distance_function
        )
    return sum_observed / (total_num_annotations_x + total_num_annotations_y)


class XrrProcessor:
    """Helper static methods for computing xRR."""

    @staticmethod
    def compute_xrr_with_summary_datasets(
        dataset1: SummaryDataset[ItemT, AnnotationT],
        dataset2: SummaryDataset[ItemT, AnnotationT],
        distance_function: DistanceFunction[AnnotationT],
        metric: XrrMetrics,
    ) -> float:
        """Compute xRR using summary-formatted datasets."""
        return metric.compute_xrr(dataset1, dataset2, distance_function)

    @staticmethod
    def compute_xrr_with_raw_datasets(
        dataset1: Iterable[tuple[ItemT, AnnotationT]],
        dataset2: Iterable[tuple[ItemT, AnnotationT]],
        distance_function: DistanceFunction[AnnotationT],
        metric: XrrMetrics,
    ) -> float:
        """Compute xRR using raw-formatted datasets."""
        return XrrProcessor.compute_xrr_with_summary_datasets(
            convert_raw_to_summary_dataset(dataset1),
            convert_raw_to_summary_dataset(dataset2),
            distance_function,
            metric,
        )


def convert_raw_to_summary_dataset(
    dataset: Iterable[tuple[ItemT, AnnotationT]],
) -> dict[ItemT, dict[AnnotationT, int]]:
    """Convert raw-formatted dataset to summary-formatted dataset."""
    summary: MutableMapping[ItemT, Counter[AnnotationT]] = defaultdict(Counter)
    for entry in dataset:
        item, annotation = entry
        summary[item][annotation] += 1
    return {item: dict(counter) for item, counter in summary.items()}


# Java-compatible aliases at module level.
compute_xrr_with_summary_datasets = XrrProcessor.compute_xrr_with_summary_datasets
compute_xrr_with_raw_datasets = XrrProcessor.compute_xrr_with_raw_datasets


__all__ = [
    "DistanceFunctions",
    "XrrMetrics",
    "XrrProcessor",
    "convert_raw_to_summary_dataset",
    "compute_xrr_with_summary_datasets",
    "compute_xrr_with_raw_datasets",
]
