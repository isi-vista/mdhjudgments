SHELL=/usr/bin/env bash
CONDA_ENV := $(shell grep '^name: ' environment.yaml | cut --delimiter ' ' --fields 2-)

.PHONY: default
default:
	@echo "an explicit target is required"

SHELL=/usr/bin/env bash

PYTHON_FILES=$(shell git ls-files '*.py' | sort | tr '\n' ' ')

export PYTHONPATH := $(shell realpath .)

.PHONY: lock
lock:
	conda-lock lock --check-input-hash --file environment.yaml --lockfile conda-lock.yaml

.PHONY: actionlint
actionlint:
	pre-commit run --all-files actionlint

.PHONY: black
black:
	pre-commit run --all-files black

.PHONY: codespell
codespell:
	pre-commit run --all-files codespell

.PHONY: lychee
lychee:
	pre-commit run --all-files --hook-stage manual lychee

.PHONY: markdownlint
markdownlint:
	pre-commit run --all-files markdownlint

.PHONY: mypy
mypy:
ifneq ($(PYTHON_FILES),)
	mypy $(PYTHON_FILES)
endif

.PHONY: prettier
prettier:
	pre-commit run --all-files prettier

.PHONY: pylint
pylint:
ifneq ($(PYTHON_FILES),)
	pylint $(PYTHON_FILES)
endif

.PHONY: ruff
ruff:
	pre-commit run --all-files ruff

.PHONY: shellcheck
shellcheck:
	pre-commit run --all-files shellcheck

.PHONY: shfmt
shfmt:
	pre-commit run --all-files shfmt

.PHONY: yamllint
yamllint:
	pre-commit run --all-files yamllint

.PHONY: precommit
precommit:
	pre-commit run --all-files

.PHONY: check
# TODO: Enable more checks
# check: precommit mypy pylint
check: precommit

.PHONY: fix
fix: lock check

UNAME=$(shell uname)
install:
	@if conda env list | grep '^${CONDA_ENV} '; then \
	  conda env remove -y -n ${CONDA_ENV}; \
	fi
	conda env create -f environment.yaml -n ${CONDA_ENV}
