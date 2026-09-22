"""
Unicode cleanup for the training text.

This module changes string length, which invalidates character offsets. It is
therefore safe on `train_unlabeled` ONLY, where the silver labels are derived
*after* normalization, and never on `validation` / `test`, whose gold offsets
index the raw string (CLAUDE.md section 5.2).

`normalize_split` enforces that rule rather than documenting it, because a
comment has already failed to prevent this once.
"""

import unicodedata

from VeriLLM import logger

# Typographic forms the generating models emit that make exact span matching
# fail downstream, plus invisible characters that inflate offsets.
_REPLACEMENTS = {
    "’": "'",    # right single quote
    "‘": "'",    # left single quote
    "“": '"',    # left double quote
    "”": '"',    # right double quote
    "„": '"',    # low double quote
    "‹": "'",    # single left angle quote
    "›": "'",    # single right angle quote
    " ": " ",    # non-breaking space
    "​": "",     # zero-width space
    "‍": "",     # zero-width joiner
}

# Splits whose labels are human and whose offsets must never move.
GOLD_SPLITS = frozenset({"validation", "test"})


def normalize_text(text):
    """
    Normalize Unicode form, curly quotes and invisible spaces in a string.

    Applies NFC composition, rewrites typographic quotes to ASCII ones, strips
    zero-width characters and trims surrounding whitespace.

    WARNING: this changes `len(text)`. See the module docstring.

    Args:
        text (str | None): the string to clean; `None` becomes `""`.

    Returns:
        str: the normalized string.

    Example:
        >>> normalize_text("  He said “hello”  ")
        'He said "hello"'
    """
    if text is None:
        return ""

    text = unicodedata.normalize("NFC", text)

    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)

    return text.strip()


def normalize_special_characters(example):
    """
    Apply `normalize_text` to the three text fields of one row.

    Args:
        example (dict): a row with `model_input`, `model_output_text` and
            `reference_answer`.

    Returns:
        dict: the same row with those three fields normalized.
    """
    example["model_input"] = normalize_text(example.get("model_input"))
    example["model_output_text"] = normalize_text(example.get("model_output_text"))
    example["reference_answer"] = normalize_text(example.get("reference_answer"))

    return example


def normalize_split(dataset, split_name):
    """
    Normalize a split, refusing to touch the gold ones.

    Args:
        dataset (datasets.Dataset): the split to normalize.
        split_name (str): which split this is. Anything in `GOLD_SPLITS` raises.

    Returns:
        datasets.Dataset: the normalized split.

    Raises:
        ValueError: if `split_name` names a gold split. Normalizing those
            silently shifts every human-annotated character offset, which is
            unrecoverable without re-downloading.

    Example:
        >>> normalize_split(None, "test")
        Traceback (most recent call last):
        ValueError: refusing to normalize the 'test' split: ...
    """
    if split_name in GOLD_SPLITS:
        raise ValueError(
            f"refusing to normalize the {split_name!r} split: its gold "
            f"character offsets index the raw string, and normalization "
            f"changes string length. See CLAUDE.md section 5.2."
        )

    logger.info(f"normalizing split: {split_name}")

    return dataset.map(normalize_special_characters)
