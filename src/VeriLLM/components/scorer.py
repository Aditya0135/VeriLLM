"""
The official Mu-SHROOM metrics.

Ported to match the shared task's own `participant_kit/scorer.py` line for
line, including its edge-case conventions. Pure functions over records - no
model, no config, no IO - so the numbers this produces are comparable to the
published leaderboard and to the paper's Table 4.

Record shape, both sides:

    reference  {"id", "lang", "text", "text_len", "hard_labels", "soft_labels"}
    prediction {"id", "hard_labels", "soft_labels"}

`hard_labels` is `[[start, end], ...]` and `soft_labels` is
`[{"start", "end", "prob"}, ...]` - the official format, never per-character
arrays (CLAUDE.md section 5.1).

Report the **macro average over languages**, not the pooled mean: `test` is
unbalanced (150 rows for the ten main languages, ~100 for the four surprise
ones), so pooling under-weights exactly the hardest rows.
"""

from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr


def score_iou(ref_dict, pred_dict):
    """
    Official IoU between gold and predicted hard labels, for one datapoint.

    Both label lists are expanded into sets of character indices; the score is
    the size of their intersection over the size of their union. The special
    case matters more than it looks: 21 of the 499 validation items have no
    gold hallucination at all, and on those an empty prediction scores 1.0
    while any prediction at all scores 0.0.

    Args:
        ref_dict (dict): gold record with `id` and `hard_labels`.
        pred_dict (dict): prediction with the same `id` and `hard_labels`.

    Returns:
        float: IoU in [0, 1]; 1.0 when gold and prediction are both empty.

    Example:
        >>> ref = {"id": "val-en-1", "hard_labels": [[0, 5]]}
        >>> pred = {"id": "val-en-1", "hard_labels": [[3, 8]]}
        >>> round(score_iou(ref, pred), 3)
        0.25
    """
    assert ref_dict["id"] == pred_dict["id"]

    ref_indices = {idx for span in ref_dict["hard_labels"] for idx in range(*span)}
    pred_indices = {idx for span in pred_dict["hard_labels"] for idx in range(*span)}

    if not pred_indices and not ref_indices:
        return 1.0

    return len(ref_indices & pred_indices) / len(ref_indices | pred_indices)


def score_cor(ref_dict, pred_dict):
    """
    Official Spearman correlation between gold and predicted soft labels.

    Both sides are expanded into dense per-character probability vectors of
    length `ref_dict["text_len"]`, then correlated. A constant vector has no
    variance, so rho is undefined; the official scorer resolves that by
    returning 1.0 when *both* vectors are constant and 0.0 otherwise, which is
    reproduced here exactly, rounding to 8 decimals to decide constancy.

    This is the metric that punishes a model trained only on hard labels: with
    nothing optimising calibration, the predicted vector tends to be nearly
    binary and correlates poorly with graded human disagreement.

    Args:
        ref_dict (dict): gold record with `id`, `text_len`, `soft_labels`.
        pred_dict (dict): prediction with the same `id` and `soft_labels`.

    Returns:
        float: Spearman rho, or the 0.0/1.0 fallback for constant vectors.
        Cast to a plain `float` - `spearmanr` returns `np.float64`, which
        `json.dump` refuses when the metrics are written out.

    Example:
        >>> ref = {"id": "x", "text_len": 4,
        ...        "soft_labels": [{"start": 0, "end": 2, "prob": 1.0}]}
        >>> pred = {"id": "x",
        ...         "soft_labels": [{"start": 0, "end": 2, "prob": 0.8}]}
        >>> round(score_cor(ref, pred), 3)
        1.0
    """
    assert ref_dict["id"] == pred_dict["id"]

    ref_vec = [0.0] * ref_dict["text_len"]
    pred_vec = [0.0] * ref_dict["text_len"]

    for span in ref_dict["soft_labels"]:
        for idx in range(span["start"], span["end"]):
            ref_vec[idx] = span["prob"]

    for span in pred_dict["soft_labels"]:
        for idx in range(span["start"], span["end"]):
            pred_vec[idx] = span["prob"]

    # constant series (i.e. no hallucination) => correlation is undefined
    if (
        len({round(flt, 8) for flt in pred_vec}) == 1
        or len({round(flt, 8) for flt in ref_vec}) == 1
    ):
        return float(
            len({round(flt, 8) for flt in ref_vec})
            == len({round(flt, 8) for flt in pred_vec})
        )

    return float(spearmanr(ref_vec, pred_vec).correlation)


def score_fbeta(ref_dict, pred_dict, beta=1.0):
    """
    Character-level precision, recall and F-beta for one datapoint.

    Not part of the official metric set - included because IoU alone cannot say
    *why* a system scores badly. Precision below recall means the system marks
    too much text; the reverse means it misses hallucinations.

    Args:
        ref_dict (dict): gold record with `id` and `hard_labels`.
        pred_dict (dict): prediction with the same `id` and `hard_labels`.
        beta (float): recall weight; 1.0 gives the harmonic mean.

    Returns:
        tuple[float, float, float]: (precision, recall, f_beta). All 1.0 when
        gold and prediction are both empty, all 0.0 when exactly one is.

    Example:
        >>> ref = {"id": "x", "hard_labels": [[0, 10]]}
        >>> pred = {"id": "x", "hard_labels": [[0, 5]]}
        >>> [round(v, 3) for v in score_fbeta(ref, pred, beta=2.0)]
        [1.0, 0.5, 0.556]
    """
    assert ref_dict["id"] == pred_dict["id"]

    ref_indices = {idx for span in ref_dict["hard_labels"] for idx in range(*span)}
    pred_indices = {idx for span in pred_dict["hard_labels"] for idx in range(*span)}

    if not ref_indices and not pred_indices:
        return 1.0, 1.0, 1.0

    if not ref_indices or not pred_indices:
        return 0.0, 0.0, 0.0

    true_positive = len(ref_indices & pred_indices)

    precision = true_positive / len(pred_indices)
    recall = true_positive / len(ref_indices)

    if precision + recall == 0.0:
        return precision, recall, 0.0

    beta_squared = beta ** 2

    f_beta = (
        (1 + beta_squared) * precision * recall
        / (beta_squared * precision + recall)
    )

    return precision, recall, f_beta


def evaluate(references, predictions, betas=(1.0, 2.0)):
    """
    Mean IoU, rho and F-beta over a set of datapoints.

    Predictions are matched to references by `id`, not by position, so the two
    lists may arrive in different orders.

    Args:
        references (list[dict]): gold records.
        predictions (list[dict]): prediction records.
        betas (tuple[float]): which F-betas to report.

    Returns:
        dict: `{"n", "iou", "cor", "precision", "recall", "f1", "f2", ...}`.

    Raises:
        KeyError: if a reference has no matching prediction.
    """
    predictions_by_id = {pred["id"]: pred for pred in predictions}

    ious = []
    cors = []
    precisions = []
    recalls = []
    fbetas = {beta: [] for beta in betas}

    for reference in references:

        prediction = predictions_by_id[reference["id"]]

        ious.append(score_iou(reference, prediction))
        cors.append(score_cor(reference, prediction))

        for beta in betas:
            precision, recall, f_beta = score_fbeta(reference, prediction, beta)
            fbetas[beta].append(f_beta)

            if beta == betas[0]:
                precisions.append(precision)
                recalls.append(recall)

    scores = {
        "n": len(references),
        "iou": float(np.mean(ious)),
        "cor": float(np.mean(cors)),
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
    }

    for beta in betas:
        scores[f"f{beta:g}"] = float(np.mean(fbetas[beta]))

    return scores


def evaluate_by_language(references, predictions, betas=(1.0, 2.0)):
    """
    Run `evaluate` separately for each language.

    Args:
        references (list[dict]): gold records carrying `lang`.
        predictions (list[dict]): prediction records.
        betas (tuple[float]): which F-betas to report.

    Returns:
        dict: `{lang: scores}`, plus a `"macro"` entry - the unweighted mean
        over languages, which is what is comparable to the paper's Table 4.
    """
    predictions_by_id = {pred["id"]: pred for pred in predictions}

    grouped = defaultdict(list)

    for reference in references:
        grouped[reference["lang"]].append(reference)

    results = {}

    for lang in sorted(grouped):
        subset = grouped[lang]
        results[lang] = evaluate(
            subset,
            [predictions_by_id[ref["id"]] for ref in subset],
            betas,
        )

    languages = [lang for lang in results]

    macro = {"n": len(languages)}

    for key in results[languages[0]]:

        if key == "n":
            continue

        macro[key] = float(np.mean([results[lang][key] for lang in languages]))

    results["macro"] = macro

    return results


def baseline_mark_none(references):
    """
    Predict no hallucination anywhere.

    Scores well only on the items with no gold hallucination. The data is
    biased toward hallucinated items, so this floor is very low.

    Args:
        references (list[dict]): gold records.

    Returns:
        list[dict]: empty predictions, one per reference.
    """
    return [
        {"id": reference["id"], "hard_labels": [], "soft_labels": []}
        for reference in references
    ]


def baseline_mark_all(references):
    """
    Predict that the entire answer is hallucinated.

    A stronger baseline than it sounds - the paper reports it around 0.345 IoU,
    above a number of submitted systems. Any system that does not beat this has
    not learned anything about *where* hallucinations are.

    Args:
        references (list[dict]): gold records carrying `text_len`.

    Returns:
        list[dict]: full-span predictions, one per reference.
    """
    return [
        {
            "id": reference["id"],
            "hard_labels": [[0, reference["text_len"]]],
            "soft_labels": [
                {"start": 0, "end": reference["text_len"], "prob": 1.0}
            ],
        }
        for reference in references
    ]


def print_scores(title, scores):
    """Print one `evaluate` result as a single aligned block."""
    print(f"{title}  (n={scores['n']})")
    print(
        f"  IoU {scores['iou']:.3f}   rho {scores['cor']:.3f}   "
        f"P {scores['precision']:.3f}   R {scores['recall']:.3f}   "
        f"F1 {scores['f1']:.3f}"
    )


def print_language_table(results_by_language):
    """
    Print per-language scores with the macro average last.

    Args:
        results_by_language (dict): output of `evaluate_by_language`.
    """
    header = f"{'lang':<8}{'n':>5}{'IoU':>8}{'rho':>8}{'P':>8}{'R':>8}{'F1':>8}"
    print(header)
    print("-" * len(header))

    for lang in sorted(k for k in results_by_language if k != "macro"):
        s = results_by_language[lang]
        print(
            f"{lang:<8}{s['n']:>5}{s['iou']:>8.3f}{s['cor']:>8.3f}"
            f"{s['precision']:>8.3f}{s['recall']:>8.3f}{s['f1']:>8.3f}"
        )

    s = results_by_language["macro"]
    print("-" * len(header))
    print(
        f"{'macro':<8}{s['n']:>5}{s['iou']:>8.3f}{s['cor']:>8.3f}"
        f"{s['precision']:>8.3f}{s['recall']:>8.3f}{s['f1']:>8.3f}"
    )
