# Repository Guidelines

## Project Structure & Module Organization

The Python 3.12 package lives in `mdhjudgments/`. Its modules cover data models, agreement and scoring utilities, hallucination analyses, and HTML review combiners. Supporting documentation for released annotations is in `mdhjudgments/SUPPLEMENTAL_DATA_README.md`. Repository-wide tooling is configured by `pyproject.toml`, `.pre-commit-config.yaml`, and the `Makefile`; dependencies are declared in `environment.yaml`. There is currently no dedicated `tests/` directory or tracked data directory. Keep generated reports, private annotations, and large analysis outputs out of version control.

## Build, Test, and Development Commands

- `make install`: recreate the `mdhjudgments` Conda environment from `environment.yaml`.
- `make precommit`: run every standard pre-commit hook against all tracked files.
- `make check`: run the repository validation target; it currently delegates to `precommit`.
- `make mypy` / `make pylint`: type-check or lint all tracked Python files.
- `make black`, `make ruff`, or `make markdownlint`: run an individual formatting or linting hook.
- `python -m mdhjudgments.score_p_r_f1 --help`: inspect a script's CLI before running it on annotation data. Use the same module form for other executable analysis modules.

## Coding Style & Naming Conventions

Use four spaces for Python and two spaces for YAML, Markdown, and other supported text formats; Makefile recipes require tabs. Black and Ruff enforce a 100-character Python line length, sorted imports, and Google-style docstrings. Mypy runs in strict mode. Name modules, functions, and variables in `snake_case`, classes in `PascalCase`, and constants in `UPPER_SNAKE_CASE`. Add type annotations to new public and internal functions.

Avoid type-printing helper functions. Prefer to write formatting logic inline.

## Testing Guidelines

No automated test suite or coverage threshold is configured yet. For every change, run `make check` plus the relevant analysis command on representative input. When adding tests, create a `tests/` directory and use unittest-compatible names such as `test_score_p_r_f1.py`; the pre-commit configuration enforces `test*.py` naming.

## Commit & Pull Request Guidelines

History currently contains only `Initial commit`, so no detailed convention is established. Use short, imperative subjects focused on the behavior changed (for example, `Fix section-level agreement calculation`, `Fix response-level Krippendorff alpha calculation` and `Add matplotlib requirement`). Keep each commit focused. Pull requests should explain the change and its data assumptions, list validation commands and results, and link the relevant issue. Include sample output or screenshots when HTML review pages or generated figures change, without exposing sensitive annotation data.

When creating a Git commit, include this trailer at the end of the commit message:

```text
Co-Authored-By: <model name> <noreply@example.com>
```
