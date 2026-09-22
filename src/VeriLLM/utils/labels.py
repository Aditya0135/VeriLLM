"""
The single owner of label conversion.

Two representations are in play throughout this project (CLAUDE.md section 5.1):

* **official** - a list of spans. `hard_labels` is `[[start, end], ...]` and
  `soft_labels` is `[{"start", "end", "prob"}, ...]`, both end-exclusive,
  both indexing **characters** of `model_output_text`. This is what the gold
  splits store and what the scorer reads.
* **per-character** - one value per character, the shape the aggregation and
  the tokenizer actually compute in.

Aggregation, tokenization and evaluation all need to cross between them. They
import from here rather than each keeping a copy, because three copies is how
an array silently reaches something expecting spans.
"""

import numpy as np

from VeriLLM.constants import OFFICIAL_CUTOFF

# Typographic variants that annotator models routinely re-type as ASCII, which
# is enough to make an otherwise verbatim span fail an exact match.
_QUOTE_MAP = {
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "„": '"', "‹": "'", "›": "'", " ": " ",
}


def _normalize_for_match(text):
    """
    Build a match-friendly copy of `text` plus a map back to original offsets.

    Collapses whitespace runs to one space, folds typographic quotes to ASCII
    and lowercases. Every character of the result maps to exactly one character
    of the input, so a match found here can be reported in original offsets.

    Lowercasing is applied per character and skipped when it would change
    length (German sharp s, dotted capital I), which is what keeps the map 1:1.

    Args:
        text (str): the original string.

    Returns:
        tuple[str, list[int]]: the normalized string, and `index_map` where
        `index_map[i]` is the index in `text` of normalized character `i`.
    """
    out = []
    index_map = []
    previous_was_space = False

    for index, character in enumerate(text):

        character = _QUOTE_MAP.get(character, character)

        if character.isspace():

            if previous_was_space:
                continue

            out.append(" ")
            index_map.append(index)
            previous_was_space = True
            continue

        lowered = character.lower()

        out.append(lowered if len(lowered) == 1 else character)
        index_map.append(index)
        previous_was_space = False

    return "".join(out), index_map


def find_span_occurrences(text, span, all_occurrences=True, fuzzy_fallback=True):
    """
    Locate an annotator's span string inside the answer.

    Fixes the two silent losses of the original `text.find(span)` approach
    (CLAUDE.md section 5.3), measured over 41,601 spans: 1.67% of spans occur
    more than once and only the first was labelled, and 5.98% failed to match
    at all because the model re-typed a quote or altered spacing.

    Exact matching is tried first and, only if it finds nothing, the
    whitespace/quote/case-normalized fallback runs. A span is never matched
    both ways, so the fallback can add recall but cannot move an exact hit.

    Args:
        text (str): the answer the span was copied from.
        span (str): the span string returned by an annotator model.
        all_occurrences (bool): label every occurrence, not just the first.
        fuzzy_fallback (bool): try the normalized match when exact fails.

    Returns:
        list[tuple[int, int]]: `(start, end)` pairs in **original** offsets,
        end-exclusive. Empty when the span could not be located at all.

    Example:
        >>> find_span_occurrences("a cat and a cat", "a cat")
        [(0, 5), (10, 15)]
    """
    if not span:
        return []

    # `find` already tolerated surrounding whitespace by never matching it;
    # stripping makes that explicit rather than accidental.
    span = span.strip()

    if not span:
        return []

    matches = []
    start = text.find(span)

    while start != -1:
        matches.append((start, start + len(span)))

        if not all_occurrences:
            break

        start = text.find(span, start + 1)

    if matches or not fuzzy_fallback:
        return matches

    # Exact matching failed - retry in normalized space.
    normalized_text, index_map = _normalize_for_match(text)
    normalized_span, _ = _normalize_for_match(span)

    if not normalized_span:
        return []

    start = normalized_text.find(normalized_span)

    while start != -1:
        end = start + len(normalized_span)

        # map the half-open normalized range back to original offsets
        matches.append((index_map[start], index_map[end - 1] + 1))

        if not all_occurrences:
            break

        start = normalized_text.find(normalized_span, start + 1)

    return matches


def spans_to_character_labels(text, spans, all_occurrences=True, fuzzy_fallback=True):
    """
    Turn one annotator's span strings into a per-character 0/1 array.

    Args:
        text (str): the answer the spans were copied from.
        spans (list[str]): span strings from a single vote.
        all_occurrences (bool): see `find_span_occurrences`. Pass False to
            reproduce the original first-occurrence-only behaviour.
        fuzzy_fallback (bool): see `find_span_occurrences`.

    Returns:
        numpy.ndarray: `int8` array of length `len(text)`, 1 where hallucinated.

    Example:
        >>> spans_to_character_labels("abc def", ["def"]).tolist()
        [0, 0, 0, 0, 1, 1, 1]
    """
    labels = np.zeros(len(text), dtype=np.int8)

    for span in spans:

        for start, end in find_span_occurrences(
            text, span, all_occurrences, fuzzy_fallback
        ):
            labels[start:end] = 1

    return labels


def labels_to_spans(text, soft_char_labels, threshold=0.5):
    """
    Convert per-character soft labels into the official Mu-SHROOM span format.

    Both outputs come from the same array, so they can never disagree with each
    other. `soft_labels` run-length encodes every non-zero run; `hard_labels`
    is the same array binarised at `threshold` and merged into contiguous spans.

    Zero-probability runs are omitted: the official scorer initialises its
    vector to 0.0 and fills in only the segments it is given, so dropping them
    is equivalent and far smaller on disk.

    Args:
        text (str): the answer the labels index into; used to validate length.
        soft_char_labels (Sequence[float]): one probability per character.
        threshold (float): hard-label cutoff, compared with `>=`.

    Returns:
        dict: `{"soft_labels": [{start, end, prob}, ...],
                "hard_labels": [[start, end], ...]}`

    Raises:
        ValueError: if the label array length does not match the text length.

    Example:
        >>> out = labels_to_spans("abcdef", [0.0, 0.8, 0.8, 0.0, 0.5, 0.5])
        >>> out["hard_labels"]
        [[1, 3], [4, 6]]
    """
    if len(soft_char_labels) != len(text):
        raise ValueError(
            f"label/text length mismatch: {len(soft_char_labels)} labels "
            f"for {len(text)} characters"
        )

    length = len(soft_char_labels)

    soft_labels = []
    run_start = 0

    for index in range(1, length + 1):

        if index == length or soft_char_labels[index] != soft_char_labels[run_start]:

            prob = float(soft_char_labels[run_start])

            if prob > 0.0:
                soft_labels.append({
                    "start": run_start,
                    "end": index,
                    "prob": prob,
                })

            run_start = index

    hard_labels = []
    span_start = None

    for index, prob in enumerate(soft_char_labels):

        if prob >= threshold and span_start is None:
            span_start = index

        elif prob < threshold and span_start is not None:
            hard_labels.append([span_start, index])
            span_start = None

    if span_start is not None:
        hard_labels.append([span_start, length])

    return {
        "soft_labels": soft_labels,
        "hard_labels": hard_labels,
    }


def spans_to_character_array(text, hard_labels):
    """
    Expand official `[start, end]` spans back into a per-character 0/1 array.

    The exact inverse of the `hard_labels` half of `labels_to_spans`. Ends are
    clamped to the text length defensively.

    Args:
        text (str): the answer the spans index into.
        hard_labels (list): `[[start, end], ...]`, end-exclusive.

    Returns:
        numpy.ndarray: `int64` array of length `len(text)`.

    Example:
        >>> spans_to_character_array("abcdef", [[1, 3]]).tolist()
        [0, 1, 1, 0, 0, 0]
    """
    labels = np.zeros(len(text), dtype=np.int64)

    for start, end in hard_labels:
        labels[int(start):min(int(end), len(text))] = 1

    return labels


def spans_to_soft_array(text, soft_labels):
    """
    Expand official `soft_labels` segments into a per-character probability array.

    Characters covered by no segment stay at 0.0, which matches the official
    scorer - it initialises its vectors to zero and fills in what it is given.

    Args:
        text (str): the answer the segments index into.
        soft_labels (Sequence[dict]): `[{"start", "end", "prob"}, ...]`.

    Returns:
        numpy.ndarray: `float32` array of length `len(text)`.

    Example:
        >>> spans_to_soft_array("abcd", [{"start": 0, "end": 2, "prob": 0.75}]).tolist()
        [0.75, 0.75, 0.0, 0.0]
    """
    probs = np.zeros(len(text), dtype=np.float32)

    for span in soft_labels:
        start = int(span["start"])
        end = min(int(span["end"]), len(text))
        probs[start:end] = float(span["prob"])

    return probs


def char_probs_to_hard_labels(char_probs, cutoff=OFFICIAL_CUTOFF):
    """
    Binarize a per-character probability vector into official `[start, end]` spans.

    Note the strict `>` comparison - it mirrors the shared task's own rule in
    `scorer.py`, and differs from the `>=` used to binarise silver labels.

    Args:
        char_probs (Sequence[float]): one probability per character.
        cutoff (float): threshold above which a character is hallucinated.

    Returns:
        list[list[int]]: `[[start, end], ...]`, end-exclusive, possibly empty.

    Example:
        >>> char_probs_to_hard_labels([0.1, 0.9, 0.8, 0.2, 0.7])
        [[1, 3], [4, 5]]
    """
    spans = []
    start = None

    for index, prob in enumerate(char_probs):

        if prob > cutoff and start is None:
            start = index

        elif prob <= cutoff and start is not None:
            spans.append([start, index])
            start = None

    if start is not None:
        spans.append([start, len(char_probs)])

    return spans


def char_probs_to_soft_labels(char_probs):
    """
    Run-length encode a probability vector into official `soft_labels` segments.

    Args:
        char_probs (Sequence[float]): one probability per character.

    Returns:
        list[dict]: `{"start": int, "end": int, "prob": float}` segments.

    Example:
        >>> char_probs_to_soft_labels([0.0, 0.5, 0.5, 0.0, 1.0])[0]["prob"]
        0.5
    """
    labels = []
    length = len(char_probs)
    start = 0

    for index in range(1, length + 1):

        if index == length or char_probs[index] != char_probs[start]:

            prob = float(char_probs[start])

            if prob > 0.0:
                labels.append({
                    "start": start,
                    "end": index,
                    "prob": prob,
                })

            start = index

    return labels
