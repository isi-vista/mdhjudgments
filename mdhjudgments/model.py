"""Data models for annotations, questions, and users."""

import datetime
from enum import StrEnum
from itertools import chain
from typing import Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _get_uuid() -> str:
    """Get a UUID represented as a string."""
    return str(uuid4())


def _now() -> datetime.datetime:
    """Gets the current UTC date and time.

    Returns:
        A timezone-aware datetime.
    """
    return datetime.datetime.now(datetime.UTC)


# Types of Incorrect Annotations
class IncorrectAnnotationType(StrEnum):
    """Types of Incorrect Annotations for information accuracy."""

    FACTUALLY_CORRECT = "correct"
    FACTUALLY_INCORRECT = "incorrect"
    NO_FACT = "not_claim"
    DISAGREEMENT = "disagreement"


# annotations of claims, citations, questions, answers
class AnnotationModel(BaseModel):
    """Base class for annotations."""

    id: str = Field(default_factory=_get_uuid)
    parent_id: str
    annotator_id: str
    timestamp: datetime.datetime = Field(default_factory=_now)


class BinaryElementCorrectnessModel(BaseModel):
    """Binary element correctness model for various topics of interest.

    Elements are True if that concept is represented in the annotation, False otherwise.
    """

    certainty: bool = False
    risk: bool = True
    urgency: bool = True


class InformationAccuracyAnnotationModel(AnnotationModel):
    """Refined section annotation model."""

    comment: str
    accuracy_type: IncorrectAnnotationType
    element_correctness: BinaryElementCorrectnessModel


class QuestionAnnotationModel(AnnotationModel):
    """Question annotation model."""

    nonsensical: bool
    dontunderstand: bool


class DimensionModel(BaseModel):
    """Dimension model."""

    model_config = ConfigDict(populate_by_name=True)

    accuracy: int | None = Field(alias="Accuracy", le=3, ge=0)

    @property
    def is_annotated(self) -> bool:
        return self.accuracy is not None


class AnswerAnnotationModel(AnnotationModel):
    """Answer annotation model."""

    dimensions: DimensionModel | None = None
    sources: list[str] = Field(default_factory=list)


class AdjudicationJudgment(StrEnum):
    """Represents the adjudicator's judgment."""

    NOT_SET = "not_set"
    HALLUCINATION = "hallucination"
    MAJOR = "major"
    MINOR = "minor"
    NO_HALLUCINATION = "no_hallucination"
    NEEDS_RESEARCH = "needs_research"
    WITH_COMMENTS_MAJOR = "with_comments_major"
    WITH_COMMENTS_MINOR = "with_comments_minor"
    WITH_COMMENTS_NO_HALLUCINATION = "with_comments_no_hallucination"


class AdjudicationCommentCategory(StrEnum):
    """Represents the adjudicator's judgment."""

    CITATIONS = "Citations"
    OTHER_MEDICAL_INFO = "Other Medical Info"
    NUMERIC_INFO = "Numeric Info"
    MISSING_INFO = "Missing Info"
    CLARITY = "Clarity"
    OTHER = "Other"


class FactCheckingCommentCategory(BaseModel):
    """Represents the factchecker's judgment of comment category."""

    simple_category: AdjudicationCommentCategory
    other_text: str | None

    @model_validator(mode="after")
    def valid_other_text(self) -> Self:
        if self.simple_category == AdjudicationCommentCategory.OTHER:
            if self.other_text:
                return self
            else:
                raise ValueError("Need other_text when comment category is other, but got None.")
        else:
            if self.other_text is None:
                return self
            else:
                raise ValueError(
                    f"Category is not other but got non-None other_text: {self.other_text!r}"
                )


class SectionAdjudicationModel(AnnotationModel):
    """Section adjudication model."""

    aggregate_judgment: AdjudicationJudgment
    q1_response: AdjudicationJudgment
    # List of categories marked for each comment
    q2_category_response: dict[str, list[AdjudicationCommentCategory]]
    #  q3a/q3b, depending on the value of q1
    #  - if q1 was "needs research", this will be q3a
    #  - if q1 was no_hallucination this will be q3b
    #  - otherwise this is not meaningful and should be set to None
    q3_response: AdjudicationJudgment
    saw_annotation_ids: list[str]
    comment: str

    @field_validator("aggregate_judgment", mode="after")
    @staticmethod
    def valid_aggregate_judgment(value: AdjudicationJudgment) -> AdjudicationJudgment:
        if value in {AdjudicationJudgment.HALLUCINATION, AdjudicationJudgment.NO_HALLUCINATION}:
            return value
        else:
            raise ValueError(
                f"Aggregate judgment must be one of 'hallucination' or 'no_hallucination'. Got: {value}"
            )

    @field_validator("q1_response", mode="after")
    @staticmethod
    def valid_q1(value: AdjudicationJudgment) -> AdjudicationJudgment:
        if value in {
            AdjudicationJudgment.MAJOR,
            AdjudicationJudgment.MINOR,
            AdjudicationJudgment.NO_HALLUCINATION,
            AdjudicationJudgment.NEEDS_RESEARCH,
        }:
            return value
        else:
            raise ValueError(
                f"Q1 response must be one of 'major', 'minor', 'needs_research', or 'no_hallucination'. Got: {value}"
            )

    @field_validator("q3_response", mode="after")
    @staticmethod
    def valid_q3(value: AdjudicationJudgment) -> AdjudicationJudgment:
        if value in {
            AdjudicationJudgment.NOT_SET,
            AdjudicationJudgment.WITH_COMMENTS_MAJOR,
            AdjudicationJudgment.WITH_COMMENTS_MINOR,
            AdjudicationJudgment.WITH_COMMENTS_NO_HALLUCINATION,
        }:
            return value
        else:
            raise ValueError(
                f"Q3 response must be one of 'not_set', 'with_comments_major', 'with_comments_minor', or 'no_hallucination'. Got: {value}"
            )

    @model_validator(mode="after")
    def valid_q3_given_q1(self) -> Self:
        """Confirm Q3 is valid in the context of Q1."""
        match self.q1_response, self.q3_response:
            case (
                (AdjudicationJudgment.MAJOR | AdjudicationJudgment.MINOR),
                AdjudicationJudgment.NOT_SET,
            ):
                return self

            case (
                (AdjudicationJudgment.NEEDS_RESEARCH | AdjudicationJudgment.NO_HALLUCINATION),
                (
                    AdjudicationJudgment.WITH_COMMENTS_MAJOR
                    | AdjudicationJudgment.WITH_COMMENTS_MINOR
                    | AdjudicationJudgment.WITH_COMMENTS_NO_HALLUCINATION
                ),
            ):
                return self

            case _:
                raise ValueError(
                    f"Q3 response doesn't make sense in the context of Q1 response. Got: {self.q3_response=}, {self.q1_response=}"
                )

    @model_validator(mode="after")
    def valid_aggregate_judgment_given_q1_and_q3(self) -> Self:
        """Confirm aggregate judgment is valid in the context of Q1 and Q3."""
        match self.aggregate_judgment, self.q1_response, self.q3_response:
            case (
                AdjudicationJudgment.HALLUCINATION,
                (AdjudicationJudgment.MAJOR | AdjudicationJudgment.MINOR),
                _,
            ):
                return self

            case (
                AdjudicationJudgment.HALLUCINATION,
                (AdjudicationJudgment.NEEDS_RESEARCH | AdjudicationJudgment.NO_HALLUCINATION),
                (
                    AdjudicationJudgment.WITH_COMMENTS_MAJOR
                    | AdjudicationJudgment.WITH_COMMENTS_MINOR
                ),
            ):
                return self

            case (
                AdjudicationJudgment.NO_HALLUCINATION,
                (AdjudicationJudgment.NEEDS_RESEARCH | AdjudicationJudgment.NO_HALLUCINATION),
                AdjudicationJudgment.WITH_COMMENTS_NO_HALLUCINATION,
            ):
                return self

            case _:
                raise ValueError(
                    f"Aggregate judgment response doesn't make sense in the context of Q1 and Q3 responses. Got: {self.aggregate_judgment=}, {self.q1_response=}, {self.q3_response=}"
                )


class FactCheckingUrlRelevance(StrEnum):
    """Represents a fact-checker's judgment of relevance."""

    SUPPORTS_OR_REFUTES = "supports_or_refutes"
    UNSUPPORTED_FABRICATION = "unsupported_fabrication"
    BROADLY_RELEVANT = "broadly_relevant"


class FactCheckingExcerptJudgment(StrEnum):
    """Represents a fact-checker's judgment of evidence's relevance to an excerpt from a chatbot response."""

    EVIDENCE_SHOWS_INCORRECT_INFO_IN_EXCERPT = "evidence_shows_incorrect_info_in_excerpt"
    LACK_OF_EVIDENCE_INDICATES_FABRICATION = "lack_of_evidence_indicates_fabrication"
    EVIDENCE_FULLY_SUPPORTS_EXCERPT = "evidence_fully_supports_excerpt"
    EVIDENCE_UNRELATED = "evidence_unrelated"


class FactCheckingCommentsJudgment(StrEnum):
    """Represents a fact-checker's judgment of evidence's relevance to comments on an excerpt from a chatbot response."""

    EVIDENCE_SUPPORTS_IDENTIFIED_ERROR = "evidence_supports_identified_error"
    EVIDENCE_SUPPORTS_IDENTIFIED_FABRICATION = "evidence_supports_identified_fabrication"
    EVIDENCE_CONTRADICTS_COMMENTS = "evidence_contradicts_comments"
    EVIDENCE_UNRELATED = "evidence_unrelated"


class FactCheckingContradictionReason(StrEnum):
    """Represents a fact-checker's judgment of relevance."""

    COMMENTS_NOT_ABOUT_FACTUAL_ACCURACY = "comments_not_about_factual_accuracy"
    JUDGE_INACCURACIES_TOO_MINOR_TO_MATTER = "judge_inaccurates_too_minor_to_matter"
    OTHER = "other"


class FactCheckingExplainContradiction(BaseModel):
    """Represents a fact-checker's judgment of relevance."""

    reason: FactCheckingContradictionReason
    explanation: str | None

    @model_validator(mode="after")
    def valid_explanation(self) -> Self:
        """Confirm aggregate judgment is valid in the context of Q1 and Q3."""
        if self.reason in {
            FactCheckingContradictionReason.COMMENTS_NOT_ABOUT_FACTUAL_ACCURACY,
            FactCheckingContradictionReason.JUDGE_INACCURACIES_TOO_MINOR_TO_MATTER,
        }:
            if self.explanation is None:
                return self
            else:
                raise ValueError(
                    "Explanation provided but reason does not allow for an explanation."
                )
        elif self.reason == FactCheckingContradictionReason.OTHER:
            if self.explanation:
                return self
            elif self.explanation is not None:
                raise ValueError("Explanation provided was blank")
            else:
                raise ValueError("Explanation not provided")
        else:
            raise ValueError(f"Unrecognized reason {self.reason}")


class SectionFactCheckingModel(AnnotationModel):
    """Section fact-checking model."""

    q1_response: AdjudicationJudgment
    # List of categories marked for each comment
    q2_category_response: list[FactCheckingCommentCategory]
    # List of comment IDs shown in Q2
    saw_annotation_ids: list[str]
    q3_choice: FactCheckingUrlRelevance
    q3_url: str
    q4_excerpt: str
    q5a_choice: FactCheckingExcerptJudgment
    q5b_choice: FactCheckingCommentsJudgment
    q5c_choice: FactCheckingExplainContradiction | None
    #  Q6a/Q6b/Q6c, depending on the value of Q5
    #  - if Q5 was "evidence refutes excerpt", this will be Q6a,
    #    how does evidence relate to comments and refute excerpt
    #  - if Q5 was "evidence supports excerpt", this will be Q6b,
    #    how does evidence relate to comments and support excerpt
    #  - if Q5 was "could not find evidence verifying or refuting excerpt", this will be Q6c,
    #    what kind of evidence would you have needed and why was it hard to find
    q6_comment: str
    aggregate_judgment: AdjudicationJudgment

    @field_validator("q1_response", mode="after")
    @staticmethod
    def valid_q1(value: AdjudicationJudgment) -> AdjudicationJudgment:
        if value in {
            AdjudicationJudgment.MAJOR,
            AdjudicationJudgment.MINOR,
            AdjudicationJudgment.NO_HALLUCINATION,
            AdjudicationJudgment.NEEDS_RESEARCH,
        }:
            return value
        else:
            raise ValueError(
                f"Q1 response must be one of 'major', 'minor', 'needs_research', or 'no_hallucination'. Got: {value}"
            )

    @field_validator("aggregate_judgment", mode="after")
    @staticmethod
    def valid_aggregate_judgement(value: AdjudicationJudgment) -> AdjudicationJudgment:
        if value in {
            AdjudicationJudgment.HALLUCINATION,
            AdjudicationJudgment.NO_HALLUCINATION,
        }:
            return value
        else:
            raise ValueError(
                f"Aggregate judgment must be one of 'hallucination' or 'no_hallucination'. Got: {value}"
            )


class Response(BaseModel):
    """LLM responses to evaluate."""

    id: str  # Taken from `AnswerModel.id`
    question: str
    answer: str


class HallucinationReason(StrEnum):
    """The reason that a section is considered a hallucination, not a hallucination, or not scoreable."""

    NO_LABEL_BECAUSE_TOO_FEW_HUMAN_ANNOTATIONS = "no_label_because_too_few_human_annotations"
    NO_LABEL_BECAUSE_MARKED_NOT_CLAIM = "no_label_because_marked_not_claim"
    NO_LABEL_BECAUSE_NO_HUMAN_CONSENSUS = "no_label_because_no_human_consensus"
    NO_LABEL_BECAUSE_NO_CONSENSUS = "no_label_because_no_consensus"
    LABEL_BECAUSE_ADJUDICATED = "label_because_adjudicated"
    LABEL_BECAUSE_HUMAN_AND_SYSTEM_CONSENSUS = "label_because_human_and_system_consensus"
    LABEL_BECAUSE_HUMAN_CONSENSUS = "label_because_human_consensus"
    LABEL_BECAUSE_CITATION_OR_LINK_ERROR = "label_because_citation_or_link_error"


class SectionResponseAnnotation(BaseModel):
    """Section Annotation Model for the Response objects."""

    id: str
    section: str
    adjudication: SectionAdjudicationModel | None = Field(default=None)
    fact_checking: SectionFactCheckingModel | None = Field(default=None)
    annotations: list[InformationAccuracyAnnotationModel] = Field(default_factory=list)


class ResponseAnnotationModel(BaseModel):
    """Response Annotation Model."""

    question_annotations: list[QuestionAnnotationModel] = Field(default_factory=list)
    answer_annotations: list[AnswerAnnotationModel] = Field(default_factory=list)
    sections_with_annotations: list[SectionResponseAnnotation] = Field(default_factory=list)

    @property
    def annotators_order(self) -> list[str]:
        return [annotation.annotator_id for annotation in self.answer_annotations]

    @property
    def accuracy_annotations(self) -> list[int]:
        """Give all the accuracy annotations."""
        return [annotation.dimensions.accuracy for annotation in self.answer_annotations]

    @property
    def sources(self) -> list[str]:
        """Give all the sources for this response."""
        # I'm doing a newline split in this because sometimes annotators
        # don't correct add multiple sources, they often use a newline
        return list(
            chain(
                s
                for annotation in self.answer_annotations
                for source in annotation.sources
                for s in source.split("\n")
            )
        )


class EnhancedResponse(Response):
    """Response with additional evaluation information."""

    annotations: ResponseAnnotationModel | None = None

    @classmethod
    def from_response(cls, response: Response) -> Self:
        """Create an enhanced response from a regular response."""
        return cls(
            id=response.id,
            question=response.question,
            answer=response.answer,
        )
