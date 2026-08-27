# MDHJudgments

This repo contains the code for the EMNLP 2026 paper _Beyond Majority Vote: Multi-Perspective Adjudication for Medical Hallucination Detection_.

Parts of this repo's code (especially the HTML visualization, scoring, hallucination analyses, and simplified data format code) are AI-authored by Codex (GPT-5.5 and GPT-5.6 Sol) and human-reviewed.

## Installation

Use the provided Makefile to install this project by running the following from the project root directory (the same directory as this README). Ensure that `conda` is in `PATH` before running this command:

```shell
make install
```

If the installation process fails, is interrupted, or for any reason needs to be restarted, run `git clean -xdf --exclude=.secrets` to reset the repository's state.

## Commands

### LLM as Judge (LaJ)

We do not provide code to run the LLM as Judge system as the system is relatively simple. We provide the prompt used in appendix E.

### Analyses

#### Table 5

```bash
PYTHONPATH=. python mdhjudgments/score_p_r_f1.py \
       --annotation-file \
       data/mdhjudgments-firstpassannotations.jsonl \
       --adjudication-files \
       data/mdhjudgments-firstmedicalexpertadjudication.jsonl \
       data/mdhjudgments-secondmedicalexpertadjudication.jsonl \
       --factchecking-files \
       data/mdhjudgments-firstfactcheckeradjudication.jsonl \
       data/mdhjudgments-secondfactcheckeradjudication.jsonl \
       --pred-annotator-id=IsiHallucinationDetector \
       --min-num-human-annotations=2 \
       --bootstrap-samples=10000 \
       --bootstrap-seed=42 \
       --output-dir \
       output/reproduce-scores
# Display the scores file as a nicely formatted table
# Omit rows showing the first-pass annotator scores against FP:AG.
# Their precision and recall is trivially 100% on this answer key.
csvlook -I output/reproduce-scores/scores.csv |
  grep -v '| first_pass_annotators \+| first_pass_annotators \+|' |
  grep -F --color=never ' responses '
```

It will take a while to compute the bootstrap confidence intervals. To skip calculating confidence intervals, pass `--bootstrap-samples=0`.

Note that the data will not be in the exact order of Table 5. This presentation of the data shows LaJ vs. FP:AG at the top (row 5 in the table), then FP+ME answer key results for FP and LaJ (rows 1 and 3 in the table), then FP+FC answer key results for FP and LaJ (rows 2 and 4 in the table).

This table also includes extra information. In Table 5 we present response-level bootstrap confidence intervals. This table additionally computes section-level confidence intervals. Additionally, this table presents the number of prediction rows. For the LaJ, this is the same as the number of sections. For first-pass annotators, this is the total number of first-pass annotations being scored.

#### Table 8

```bash
for i in $(seq 0 8); do
  PYTHONPATH=. python mdhjudgments/score_p_r_f1.py \
       --annotation-file \
       data/mdhjudgments-with-gpt-and-gemini-reruns.jsonl \
       --adjudication-files \
       data/mdhjudgments-firstmedicalexpertadjudication.jsonl \
       data/mdhjudgments-secondmedicalexpertadjudication.jsonl \
       --factchecking-files \
       data/mdhjudgments-firstfactcheckeradjudication.jsonl \
       data/mdhjudgments-secondfactcheckeradjudication.jsonl \
       --pred-annotator-id=LaJDetectorRerunGPT-5."${i:?}" \
       --min-num-human-annotations=2 \
       --bootstrap-samples=0 \
       --bootstrap-seed=42 \
       --output-dir \
       output/reproduce-scores/gpt_"${i:?}"
  PYTHONPATH=. python mdhjudgments/score_p_r_f1.py \
       --annotation-file \
       data/mdhjudgments-with-gpt-and-gemini-reruns.jsonl \
       --adjudication-files \
       data/mdhjudgments-firstmedicalexpertadjudication.jsonl \
       data/mdhjudgments-secondmedicalexpertadjudication.jsonl \
       --factchecking-files \
       data/mdhjudgments-firstfactcheckeradjudication.jsonl \
       data/mdhjudgments-secondfactcheckeradjudication.jsonl \
       --pred-annotator-id=LaJDetectorRerunGemini3.5Flash."${i:?}" \
       --min-num-human-annotations=2 \
       --bootstrap-samples=0 \
       --bootstrap-seed=42 \
       --output-dir \
       output/reproduce-scores/gemini_"${i:?}"
done
mkdir -p output/reproduce-scores
printf 'answer_key,predictor,precision,precision_bootstrap95%ci_lower,precision_bootstrap95%ci_upper,recall,recall_bootstrap95%ci_lower,recall_bootstrap95%ci_upper,f1,f1_bootstrap95%ci_lower,f1_bootstrap95%ci_upper,n_answer_key_sections,n_prediction_rows,bootstrap_ci_sampledover,bootstrap_ci_samples,bootstrap_ci_seed' \
       > output/reproduce-scores/table8_scores.csv
cat output/reproduce-scores/*/scores.csv |
    tr -d '\r' |
    grep -F 'sections' |
    grep -F 'LaJDetectorRerun' >> output/reproduce-scores/table8_scores.csv
python -c '
from pathlib import Path
import sys

import pandas as pd

table8_scores_csv_path = Path(sys.argv[1])

df = pd.read_csv(table8_scores_csv_path)
gpt5_df = df[df["predictor"].str.contains("GPT-5")]
gemini_df = df[df["predictor"].str.contains("Gemini3.5Flash")]

relevant_rows = ["50%", "min", "max"]

print("Gemini-ME")
print((100 * gemini_df.loc[gemini_df["answer_key"] == "first_pass_annotators_plus_medical_experts", ["precision", "recall"]]).describe().transpose()[relevant_rows].transpose().round(0))

print("GPT-5-ME")
print((100 * gpt5_df.loc[gpt5_df["answer_key"] == "first_pass_annotators_plus_medical_experts", ["precision", "recall"]]).describe().transpose()[relevant_rows].transpose().round(0))

print("Gemini-FcCk")
print((100 * gemini_df.loc[gemini_df["answer_key"] == "first_pass_annotators_plus_fact_checkers", ["precision", "recall"]]).describe().transpose()[relevant_rows].transpose().round(0))

print("GPT-5-FcCk")
print((100 * gpt5_df.loc[gpt5_df["answer_key"] == "first_pass_annotators_plus_fact_checkers", ["precision", "recall"]]).describe().transpose()[relevant_rows].transpose().round(0))
' output/reproduce-scores/table8_scores.csv
```

#### All other analyses

```bash
mkdir -p output
PYTHONPATH=. python mdhjudgments/run_hallucinations_analyses.py \
             --annotation-file \
             data/mdhjudgments-firstpassannotations.jsonl \
             --adjudication-files \
             data/mdhjudgments-firstmedicalexpertadjudication.jsonl \
             data/mdhjudgments-secondmedicalexpertadjudication.jsonl \
             --factchecking-files \
             data/mdhjudgments-firstfactcheckeradjudication.jsonl \
             data/mdhjudgments-secondfactcheckeradjudication.jsonl \
             --medexpert-adjudication-files \
             data/medexpert-adjudicated.jsonl \
             --save-figures-to \
             output/reproduce-figures \
             --bootstrap-samples=10000 \
             --bootstrap-seed=42 |
             tee output/reproduce-analyses.md
```

In the output:

- We show Tables 1 and 2 directly.
- The information in Tables 3 and 4 are broken over several sections.
  - The agreement numbers from Table 3 (Agree:Err=True, Agree:Err=False, and Disagree) are presented as a series of separate tables for each group. The off-diagonal entries are intentionally identical --- we do not distinguish between the first and second annotator or adjudicator within a group. Thus, both off-diagonal entries give the same count for the number of disagreements.
  - The agreement numbers from Table 4 (Agr:Err=True, Agr:Err=False, Disagree:TF, and Disagree:FT) are presented as a series of separate tables for each group pairing.
  - The % agreement and Krippendorff's alpha numbers are listed under a separate "Agreement statistics" subsection.
- We show Table 6 directly.
- The information in Table 7 appears in the "Agreement statistics" subsection that also shows information from Tables 3 and 4.
- We show Tables 9, 10, and 11 directly.
- Figure 2 corresponds to two files in `reproduce-figures/`:
  - `avg_scalar_accuracy_vs_pct_labeled_hallucination_both_short_responses.pdf`, the upper part of the figure
  - `avg_scalar_accuracy_vs_pct_labeled_hallucination_both_long_responses.pdf`, the lower part of the figure
- Due to a limitation of our anonymization approach, the numbers in appendix F cannot be reproduced from the anonymized data. These include of unique medical experts/fact-checkers and the numbers of sections each adjudicator adjudicated. They can't be reproduced using the public data because the anonymized medical expert and fact-checker IDs depend both on the adjudicator's Prolific ID and upon the ordering of their response in the Qualtrics data export.

### Reproducing the human-readable data from supplemental

```bash
mkdir -p output/visualizations

# First pass annotations
PYTHONPATH=. python mdhjudgments/show_annotation_summary.py \
             data/mdhjudgments-firstpassannotations.jsonl \
             --output \
             output/visualizations/mdhjudgments-firstpassannotations.html

# Medical expert adjudications
PYTHONPATH=. python mdhjudgments/show_adjudication_summary.py \
             data/mdhjudgments-firstmedicalexpertadjudication.jsonl \
             --output \
             output/visualizations/mdhjudgments-firstmedicalexpertadjudication.html
PYTHONPATH=. python mdhjudgments/show_adjudication_summary.py \
             data/mdhjudgments-secondmedicalexpertadjudication.jsonl \
             --output \
             output/visualizations/mdhjudgments-secondmedicalexpertadjudication.html
PYTHONPATH=. python mdhjudgments/combine_adjudication_html_for_review.py \
             output/visualizations/mdhjudgments-firstmedicalexpertadjudication.html \
             output/visualizations/mdhjudgments-secondmedicalexpertadjudication.html \
             --output \
             output/visualizations/mdhjudgments-dualmedicalexpertadjudications.html
PYTHONPATH=. python mdhjudgments/show_adjudication_summary.py \
             data/medexpert-adjudicated.jsonl \
             --output \
             output/visualizations/medexpert-adjudicated.html

# Fact checker adjudications
PYTHONPATH=. python mdhjudgments/show_factchecking_summary.py \
             data/mdhjudgments-firstfactcheckeradjudication.jsonl \
             --output \
             output/visualizations/mdhjudgments-firstfactcheckeradjudication.html
PYTHONPATH=. python mdhjudgments/show_factchecking_summary.py \
             data/mdhjudgments-secondfactcheckeradjudication.jsonl \
             --output \
             output/visualizations/mdhjudgments-secondfactcheckeradjudication.html
PYTHONPATH=. python mdhjudgments/combine_factchecking_html_for_review.py \
             output/visualizations/mdhjudgments-firstfactcheckeradjudication.html \
             output/visualizations/mdhjudgments-secondfactcheckeradjudication.html \
             --output \
             output/visualizations/mdhjudgments-dualfactcheckeradjudications.html

# First pass annotations plus model judgments used in Table 8
PYTHONPATH=. python mdhjudgments/show_annotation_summary.py \
             data/mdhjudgments-with-gpt-and-gemini-reruns.jsonl \
             --output \
             output/visualizations/mdhjudgments-with-gpt-and-gemini-reruns.html
```

### Simplified data format

We provide our data in JSONL format for reproducing our analyses since this is closer to our original internal format. However, for convenience of others using this dataset for analyses, we also provide a script which reformats the data as a set of linked CSV files, run as follows:

```bash
PYTHONPATH=. python mdhjudgments/enhanced_responses_to_csv.py \
             data/mdhjudgments-firstpassannotations.jsonl \
             data/mdhjudgments-firstmedicalexpertadjudication.jsonl \
             data/mdhjudgments-secondmedicalexpertadjudication.jsonl \
             data/mdhjudgments-firstfactcheckeradjudication.jsonl \
             data/mdhjudgments-secondfactcheckeradjudication.jsonl \
             output/reformatted-data
```

The output is structured as follows:

- `questions_responses.csv` contains the response IDs and the corresponding question and response pair.
- `sections.csv` contains the section IDs and the corresponding response IDs, section text, and ordering of the sections within the response (the first section is numbered 0).
- `response_level_annotations.csv` contains the first-pass annotations of response-level accuracy: Each row contains the response ID annotated, the annotator ID, and the assigned accuracy score (on a scale of 1-3, with 3 being the most accurate and 1 being the least accurate).
- `section_level_first_pass_annotations.csv` contains the first pass annotations of section-level accuracy. Each row contains the section ID, annotation or "comment" ID, annotator ID, a convenience flag for checking if the annotator is human (is not LLM as judge), and the annotator's judgment of the section text:
  - `accuracy_type` denotes their judgment of whether the section contains a claim vs. is correct, incorrect, or contains information medical professionals would disagree about
  - `certainty`, `risk`, and `urgency` contain, respectively, the annotator's judgments about whether the section is correct with respect to its expression of certainty, risk, and urgency
    - Each of these is `true` if they think the section's expression of certainty/risk/urgency is appropriate, and `false` if they think the section's expression is inappropriate.
  - `has_hallucination` is a simplified summary of the annotator's judgment which is true if they judged the section `incorrect` (`accuracy_type`), or if they marked any of certainty, risk, or urgency as inappropriately expressed (`false`).
- `section_level_medical_expert_adjudications.csv` contains the medical expert adjudication of section-level correctness. Each row contains the section ID, medical expert ID, their responses to Q1 and Q3, the comment they provided, a JSON-encoded list of annotation IDs they reviewed, and a simplified summary of their judgment (`aggregate_judgment`). The simplified summary is computed as described in appendix section D.3.
  - Note that adjudicator IDs may differ even when the underlying adjudicator is the same. See the note under `Analyses` about anonymization.
- `comment_level_medical_expert_adjudication_judgments.csv` contains the medical expert adjudication of comment-level categories. Each row contains the first-pass annotation ID, the medical expert adjudicator ID, and one column per each of the six categories. A category column is `true` when the adjudicator judged the category as applying to the comment and `false` otherwise.
- `section_level_factchecker_adjudications.csv` contains the factchecker adjudication of section-level correctness. Each row contains the section ID, factchecker "annotator" ID, their responses to Q1, Q3, Q4, Q5a, Q5b, (if applicable) Q5c, their Q6 comment, a list of annotation IDs they saw, and one column per category they assigned to the comments they reviewed. A category column is `true` when the adjudicator judged the category as applying to the comment and `false` otherwise. In this survey unlike for medical experts we also collect a free text description for what the adjudicator meant by the `Other` category, and we provide this in the `Other text` column where applicable.
  - Note that adjudicator IDs may differ even when the underlying adjudicator is the same. See the note under `Analyses` about anonymization.

## Contributing

This project uses [pre-commit](https://pre-commit.com/), which is automatically installed with the rest of the development requirements.

Pre-commit checks can now run automatically when you make a commit. If you want to run a subset of checks for formatting `make precommit` still runs some checks.

If you want to use the git-hook to run checks automatically run `pre-commit install` and all checks will run for each commit.

## License

The code and related documentation in this repository is released under an MIT license --- see the file `LICENSE` for details.

[![CC BY 4.0][cc-by-shield]][cc-by]

Except for third-party evidence excerpts, the MDHJudgments dataset is licensed under a [Creative Commons Attribution 4.0 International License][cc-by]. Evidence excerpts remain subject to the rights of their respective source owners and are included with source URLs for research and scholarly analysis.

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

This dataset includes short excerpts from publicly available sources to preserve the evidence consulted during fact-checking adjudication. Each excerpt is accompanied by the URL of its source. If you are a rights holder and would like us to review an excerpt for removal, please contact [mrf@isi.edu](mailto:mrf@isi.edu) and identify the excerpt, its source URL, and the basis for your request. We will review reasonable requests and, where appropriate, remove or replace the excerpt in a subsequent release.

## Citation

TODO
