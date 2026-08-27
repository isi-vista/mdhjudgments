"""Script to visualize annotation data as HTML."""

import argparse
from dataclasses import dataclass
from pathlib import Path

from dominate import document
from dominate.tags import div, style, table, td, th, tr

from mdhjudgments.file_utils import read_jsonl
from mdhjudgments.model import (
    AnswerAnnotationModel,
    EnhancedResponse,
    QuestionAnnotationModel,
)


@dataclass
class QuestionEntry:
    """Represents a single question and its (response- and section-level) annotations."""

    question: str
    answer_id: str
    answer_annotations: list[AnswerAnnotationModel] | None
    question_annotations: list[QuestionAnnotationModel] | None
    sections: list[dict]


def format_text(text: str) -> str:
    """Format text so it can be used in HTML output."""
    return text.replace("\n", "<br>\n")


def build_html(
    responses: list[EnhancedResponse],
    annotator_filter: set[str] | None = None,
    gate_annotators: set[str] | None = None,
) -> str:
    """Render list of responses with section-level adjudications as HTML."""
    question_entries: list[QuestionEntry] = []
    for response in responses:
        anns = response.annotations
        if anns is None:
            continue

        gate_match = False
        if gate_annotators:
            for a in anns.answer_annotations or []:
                if a.annotator_id and a.annotator_id.lower() in gate_annotators:
                    gate_match = True
                    break
            if not gate_match:
                for sec in anns.sections_with_annotations or []:
                    for sa in sec.annotations or []:
                        if sa.annotator_id and sa.annotator_id.lower() in gate_annotators:
                            gate_match = True
                            break
                    if gate_match:
                        break

        if gate_annotators and not gate_match:
            continue

        section_list: list[dict] = []
        for section in anns.sections_with_annotations:
            annotations = section.annotations
            if len(annotations) <= 1:
                continue
            rows = []
            for a in annotations:
                if (
                    (not gate_match)
                    and annotator_filter
                    and a.annotator_id
                    and a.annotator_id.lower() not in annotator_filter
                ):
                    continue
                ec = a.element_correctness
                rows.append(
                    {
                        "annotator_id": a.annotator_id,
                        "accuracy_type": a.accuracy_type,
                        "certainty": "true" if (ec and ec.certainty) else "false",
                        "risk": "true" if (ec and ec.risk) else "false",
                        "urgency": "true" if (ec and ec.urgency) else "false",
                        "comment": a.comment,
                    }
                )
            if rows:
                section_list.append(
                    {
                        "section_id": section.id,
                        "section_text": section.section,
                        "rows": rows,
                    }
                )

        if gate_match:
            question_entries.append(
                QuestionEntry(
                    question=response.question,
                    answer_id=response.id,
                    answer_annotations=anns.answer_annotations,
                    question_annotations=anns.question_annotations,
                    sections=section_list,
                )
            )
        else:
            if annotator_filter is not None:
                filtered_answer_annotations = [
                    a
                    for a in (anns.answer_annotations or [])
                    if a.annotator_id and a.annotator_id.lower() in annotator_filter
                ]
                filtered_question_annotations = [
                    qa
                    for qa in (anns.question_annotations or [])
                    if qa.annotator_id and qa.annotator_id.lower() in annotator_filter
                ]
                include_question = (
                    bool(section_list)
                    or bool(filtered_answer_annotations)
                    or bool(filtered_question_annotations)
                )
                if include_question:
                    question_entries.append(
                        QuestionEntry(
                            question=response.question,
                            answer_id=response.id,
                            answer_annotations=filtered_answer_annotations,
                            question_annotations=filtered_question_annotations,
                            sections=section_list,
                        )
                    )
            else:
                if section_list:
                    question_entries.append(
                        QuestionEntry(
                            question=response.question,
                            answer_id=response.id,
                            answer_annotations=anns.answer_annotations,
                            question_annotations=anns.question_annotations,
                            sections=section_list,
                        )
                    )

    doc = document()

    css = """
    /* Grouping box to clearly associate question, answer, and response */
    .qa-box {
      border: 2px solid #bbb;
      border-radius: 6px;
      padding: 10px 12px;
      margin: 18px 0 24px 0;
      background: #fcfcff;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    table.mygrid, table.question-table, table.answer-table, table.section-table {
      border-collapse: collapse;
      border: 1px solid #666;
      width: 100%;
    }
    table.mygrid th, table.mygrid td, table.question-table th, table.question-table td,
    table.answer-table th, table.answer-table td, table.section-table th, table.section-table td {
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
    table.question-table th.spanning {
      background: #e9f2ff;
      border-bottom: 2px solid #2c7be5;
      font-size: 1.05rem;
      font-weight: 700;
      padding: 10px;
    }
    table.answer-table, table.section-table {
      margin: 10px 0 20px 0;
    }
    table.mygrid th.spanning, table.answer-table th.spanning, table.section-table th.spanning {
      max-width: 40ch;
      text-align: left;   /* left-justify the spanning cell */
      background: #f0f0f0;
      font-weight: bold;
      padding: 8px;
    }
    table.subgrid th {
      background: #fafafa;
    }
    """

    doc.add(style(css))

    # Total columns for the per-answer summary: annotator_id + 7 dimensions + sources/missing_info
    total_cols = 1 + 7 + 1  # = 9

    for q in question_entries:
        # Wrap question, answer, and response tables in a grouping box
        group = div(cls="qa-box")

        # 1) Question header table
        header_tbl = table(cls="question-table")
        question_html = format_text("QUESTION:\n" + q.question)

        top_tr = tr()
        top_th = th(colspan=total_cols, _class="spanning")
        top_th.add_raw_string(question_html)
        top_tr.add(top_th)
        header_tbl.add(top_tr)

        header_tbl.add(tr(td(f"Answer ID: {q.answer_id}", colspan=total_cols)))

        # Insert Question Annotations inside the question table (if any)
        qa = q.question_annotations
        if qa:
            # Subheader for question annotations
            header_tbl.add(tr(th("Question Annotations", colspan=total_cols, _class="spanning")))
            # Create headers using colspans to fit the question-table grid
            hdr = tr()
            hdr.add(th("annotator_id", colspan=3))
            hdr.add(th("nonsensical", colspan=3))
            hdr.add(th("dontunderstand", colspan=3))
            header_tbl.add(hdr)
            # Rows
            for a in qa:
                r = tr()
                r.add(td(a.annotator_id, colspan=3))
                r.add(td("true" if a.nonsensical else "false", colspan=3))
                r.add(td("true" if a.dontunderstand else "false", colspan=3))
                header_tbl.add(r)

        group.add(header_tbl)

        # 2) Per-answer annotation table (wide)
        aa = q.answer_annotations
        if aa:
            aa_tbl = table(cls="answer-table")
            aa_tbl.add(tr(th("Per-Answer Annotations", colspan=total_cols, _class="spanning")))

            sub_hdr = tr()
            for hname in [
                "annotator_id",
                "accuracy",
                "sources",
            ]:
                sub_hdr.add(th(hname))
            aa_tbl.add(sub_hdr)

            added_any_row = False
            for a in aa:
                dims = a.dimensions
                row = tr()
                row.add(td(a.annotator_id))
                row.add(td("" if dims is None else str(dims.accuracy)))
                row.add(td("\n\n".join(a.sources)))

                aa_tbl.add(row)
                added_any_row = True

            if added_any_row:
                group.add(aa_tbl)

        # 3) Separate section tables (narrow)
        for s in q.sections:
            sec_tbl = table(cls="section-table")

            section_text_html = format_text("SECTION:\n" + s["section_text"])
            top_tr = tr()
            top_th = th(colspan=7, _class="spanning")
            top_th.add_raw_string(section_text_html)
            top_tr.add(top_th)
            sec_tbl.add(top_tr)

            rows = s["rows"]
            if not rows:
                continue
            hdr_tr = tr()
            for hname in rows[0]:
                hdr_tr.add(th(hname))
            sec_tbl.add(hdr_tr)

            for row_data in rows:
                row_tr = tr()
                for v in row_data.values():
                    row_tr.add(td(v))
                sec_tbl.add(row_tr)

            group.add(sec_tbl)
        doc.body.add(group)
    return doc.render()


def main() -> None:  # noqa: D103
    parser = argparse.ArgumentParser(description="Show annotations using EnhancedResponse objects.")
    parser.add_argument("jsonl_file", type=Path, help="Path to JSONL with EnhancedResponse rows")
    parser.add_argument(
        "--output", type=Path, default=Path("disagreement.html"), help="Output HTML path"
    )
    parser.add_argument(
        "--annotators",
        nargs="+",
        default=None,
        help="Filter to these annotator names/IDs (case-insensitive). Supports space- or comma-separated values.",
    )
    parser.add_argument(
        "--gate-annotators",
        nargs="+",
        default=None,
        help=(
            "Include a question if any of these annotators has annotated it (answer- or section-level); "
            "show all annotations for that question. Supports space- or comma-separated values."
        ),
    )
    args = parser.parse_args()

    annotator_filter: set[str] | None = None
    if args.annotators:
        raw_items: list[str] = []
        for item in args.annotators:
            raw_items.extend(item.split(","))
        annotator_filter = {s.strip().lower() for s in raw_items if s.strip()}

    gate_annotators: set[str] | None = None
    if args.gate_annotators:
        raw_gates: list[str] = []
        for item in args.gate_annotators:
            raw_gates.extend(item.split(","))
        gate_annotators = {s.strip().lower() for s in raw_gates if s.strip()}

    responses = [EnhancedResponse.model_validate(r) for r in read_jsonl(args.jsonl_file)]
    html = build_html(responses, annotator_filter=annotator_filter, gate_annotators=gate_annotators)
    args.output.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
