"""
Small IO helpers shared by every stage.

Nothing here knows about Mu-SHROOM. The jsonl pair matters most: the annotation
loop is append-only and resumable, so it must never rewrite a file it is adding
to (CLAUDE.md section 0 - the caches are the expensive artifact of the project).
"""

import json
import os
from pathlib import Path

import yaml

from VeriLLM import logger


def read_yaml(path_to_yaml: Path) -> dict:
    """
    Read a YAML file.

    Args:
        path_to_yaml (Path): file to read.

    Returns:
        dict: the parsed contents.

    Raises:
        ValueError: if the file is empty. An empty config is always a mistake
            here, and failing loudly beats a `None` surfacing three stages later.
    """
    with open(path_to_yaml, "r", encoding="utf-8") as file:
        content = yaml.safe_load(file)

    logger.info(f"loaded yaml file from: {path_to_yaml}")

    if content is None:
        raise ValueError(f"YAML file {path_to_yaml} is empty.")

    return content


def create_directories(path_to_directories: list, verbose: bool = True) -> None:
    """
    `makedirs(exist_ok=True)` for each path.

    Args:
        path_to_directories (list): directories to create.
        verbose (bool): log each creation.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)

        if verbose:
            logger.info(f"created directory at: {path}")


def save_json(path: Path, data: dict) -> None:
    """
    Write JSON, indented and without ASCII escaping.

    `ensure_ascii=False` is required, not cosmetic: this project stores Arabic,
    Hindi and Chinese answers, and escaping them makes the caches unreadable.
    """
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def load_json(path: Path) -> dict:
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path) -> list:
    """
    Read one record per line, skipping blank lines.

    Returns an empty list when the file does not exist, so a resumable loop can
    call this on its first ever run.
    """
    records = []

    if not os.path.exists(path):
        return records

    with open(path, "r", encoding="utf-8") as file:

        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def append_jsonl(path: Path, record: dict) -> None:
    """
    Append ONE record.

    Opens in `"a"` mode and never truncates. This is what makes the annotation
    cache survive a crash, a rate limit or a closed laptop.
    """
    with open(path, "a", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False)
        file.write("\n")
