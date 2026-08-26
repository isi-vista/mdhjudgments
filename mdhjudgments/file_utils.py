"""Common shared logic for working with files."""

from collections.abc import Iterable
import json
from os import PathLike
from typing import Any


def read_jsonl(path: PathLike | str) -> Iterable[Any]:
    """Read data in JSON Lines format.

    Args:
        path: Path to file.
    """
    with open(path, encoding="utf-8") as jsonl_in:
        for line in jsonl_in:
            yield json.loads(line)
