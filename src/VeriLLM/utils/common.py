import json
import os
from pathlib import Path
import yaml
from VeriLLM import logger


def read_yaml(path_to_yaml: Path) -> dict:
    """Read a YAML file. Raises ValueError if it is empty."""
    with open(path_to_yaml, "r", encoding="utf-8") as file:
        content = yaml.safe_load(file)
        logger.info(f"loaded yaml file from: {path_to_yaml}")
        if content is None:
            raise ValueError(f"YAML file {path_to_yaml} is empty.")
        return content


def create_directories(path_to_directories: list, verbose: bool = True) -> None:
    """makedirs(exist_ok=True) for each path."""
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
        logger.info(f"created directory at: {path}")


def save_json(path: Path, data: dict) -> None:
    """Write JSON with indent=4, ensure_ascii=False, encoding='utf-8'."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def load_json(path: Path) -> dict:
    """Read JSON with encoding='utf-8'."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def load_jsonl(path: Path) -> list:
    """Read one record per line, skipping blank lines."""
    records = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def append_jsonl(path: Path, record: dict) -> None:
    """Append ONE record. Opens in 'a' mode - never rewrites the file."""
    with open(path, "a", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False)
        file.write("\n")
