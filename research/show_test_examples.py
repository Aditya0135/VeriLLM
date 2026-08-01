"""
Print one worked Mu-SHROOM test example per language: the input the model saw,
the answer it scored, the spans it marked as hallucinated, and the gold spans.

Standalone. Does NOT need train.ipynb to have been run - it loads the saved
checkpoint from disk and runs inference on only the handful of rows it prints,
so it finishes in seconds even on CPU.

Usage:
    python show_test_examples.py
    python show_test_examples.py --langs HI EN ES
    python show_test_examples.py --index 3
    python show_test_examples.py --model ./xlm-roberta-base-silver-with-ref-final
    python show_test_examples.py --cutoff 0.3

The encoding, projection and binarization here are copied from train.ipynb
(`encode_pair`, `predict_char_probs`, `char_probs_to_hard_labels`) so the spans
printed are the same ones the official scorer saw.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Devanagari, Arabic and CJK all die on a cp1252 Windows console otherwise.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MAX_LENGTH = 512

# Checkpoints live next to this file, not next to wherever you launched python
# from. Without this, running from the repo root makes transformers read the
# relative path as a Hub repo id and fail with HFValidationError.
SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_model(name):
    """Accept a Hub id, a path relative to the cwd, or one relative to this script."""
    for candidate in (Path(name), SCRIPT_DIR / name):
        if candidate.is_dir():
            return str(candidate.resolve())

    # not a local directory - hand it to transformers as a Hub repo id
    return name


def encode_pair(tokenizer, question, answer, answer_only=False):
    """Tokenize (question, answer) exactly as training did; offsets index the answer."""
    if answer_only:
        return tokenizer(answer, truncation=True, max_length=MAX_LENGTH,
                         return_offsets_mapping=True)
    try:
        # trims the question, never the answer - gold offsets index the answer
        return tokenizer(question, answer, truncation="only_first",
                         max_length=MAX_LENGTH, return_offsets_mapping=True)
    except Exception:
        # answer alone exceeds the budget; nothing left to take off the question
        return tokenizer(question, answer, truncation="longest_first",
                         max_length=MAX_LENGTH, return_offsets_mapping=True)


def predict_char_probs(row, model, tokenizer, device, answer_only=False):
    """One hallucination probability per character of `model_output_text`."""
    text = row["model_output_text"]
    encoding = encode_pair(tokenizer, row["model_input"], text, answer_only)

    sequence_ids = encoding.sequence_ids()
    answer_id = None if answer_only else 1

    inputs = {
        "input_ids": torch.tensor([encoding["input_ids"]]).to(device),
        "attention_mask": torch.tensor([encoding["attention_mask"]]).to(device),
    }

    with torch.no_grad():
        logits = model(**inputs).logits

    token_probs = torch.softmax(logits, dim=-1)[0, :, 1].cpu().numpy()

    char_probs = np.zeros(len(text), dtype=float)

    for index, (start, end) in enumerate(encoding["offset_mapping"]):

        # skip specials, and (in pair mode) every question token
        if start == end or sequence_ids[index] != answer_id:
            continue

        char_probs[start:end] = np.maximum(char_probs[start:end], token_probs[index])

    return char_probs


def char_probs_to_spans(char_probs, cutoff):
    """Merge runs above `cutoff` into official half-open [start, end) spans."""
    spans, start = [], None

    for index, prob in enumerate(char_probs):
        if prob > cutoff and start is None:
            start = index
        elif prob <= cutoff and start is not None:
            spans.append([start, index])
            start = None

    if start is not None:
        spans.append([start, len(char_probs)])

    return spans


def underline(text, spans):
    """Return `text` with every span underlined (ANSI SGR 4)."""
    out, cursor = [], 0

    for start, end in sorted(spans):
        out += [text[cursor:start], "\033[4m", text[start:end], "\033[24m"]
        cursor = end

    return "".join(out + [text[cursor:]])


def show(row, model, tokenizer, device, cutoff, answer_only):
    text = row["model_output_text"]

    char_probs = predict_char_probs(row, model, tokenizer, device, answer_only)
    predicted = char_probs_to_spans(char_probs, cutoff)
    gold = [[int(start), int(end)] for start, end in row["hard_labels"]]

    print("=" * 90)
    print(f"[{row['id']}]  lang={row['lang']}")

    print(f"\nMODEL INPUT:   {row['model_input']}")
    print(f"MODEL OUTPUT:  {text}")

    print(f"\nSpans detected by LLM:     {underline(text, predicted)}")
    print(f"Spans detected by Humans:  {underline(text, gold)}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="xlm-roberta-base-silver-final")
    parser.add_argument("--langs", nargs="+", default=["HI", "EN"])
    parser.add_argument("--index", type=int, default=0,
                        help="which matching row to show, 0-based")
    parser.add_argument("--cutoff", type=float, default=0.5,
                        help="official binarization threshold")
    parser.add_argument("--answer-only", action="store_true",
                        help="checkpoint was trained without the question in the input")
    parser.add_argument("--any", action="store_true",
                        help="do not skip items whose gold is empty")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = resolve_model(args.model)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path).to(device).eval()
    print(f"loaded {model_path} on {device}\n")

    test = load_dataset("Helsinki-NLP/mu-shroom", "all")["test"]

    for lang in args.langs:

        pool = [row for row in test if row["lang"] == lang.upper()]

        if not args.any:
            # a clean item makes a dull example; fall back if the language has none
            pool = [row for row in pool if row["hard_labels"]] or pool

        if not pool:
            print(f"no rows for lang={lang}")
            continue

        show(pool[args.index % len(pool)], model, tokenizer, device,
             args.cutoff, args.answer_only)


if __name__ == "__main__":
    main()
