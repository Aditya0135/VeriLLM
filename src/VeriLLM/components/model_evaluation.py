"""
Official evaluation against the gold splits.

Reports the macro average over languages, which is what is comparable to the
task paper's Table 4. `test` is unbalanced - 150 rows for the ten main
languages and ~100 for the four surprise ones - so a pooled mean under-weights
exactly the hardest rows.

The threshold sweep runs on **validation only**. Tuning on test would make the
headline number meaningless; the sweep is analysis, showing whether a weak IoU
is a calibration problem or a model problem. ATLANTIS (arXiv:2508.05179) found
their model globally under-confident and lowered the threshold to raise IoU;
our own sweep peaked at the lowest value tested.
"""

import json

import numpy as np

from VeriLLM import logger
from VeriLLM.constants import OFFICIAL_CUTOFF, SURPRISE_LANGUAGES
from VeriLLM.components.prediction import build_prediction_records
from VeriLLM.components.scorer import (
    baseline_mark_all,
    baseline_mark_none,
    evaluate,
    evaluate_by_language,
    print_language_table,
    print_scores,
)
from VeriLLM.entity.config_entity import ModelEvaluationConfig


def sweep_cutoff(references, all_char_probs, cutoffs, betas=(1.0, 2.0)):
    """
    Score the same probability vectors at a range of decision thresholds.

    Args:
        references (list[dict]): gold records.
        all_char_probs (list[numpy.ndarray]): predicted probabilities.
        cutoffs (Sequence[float]): thresholds to try.
        betas (tuple): F-betas to report.

    Returns:
        dict: `{cutoff: scores}`.
    """
    results = {}

    for cutoff in cutoffs:
        predictions = build_prediction_records(
            references, all_char_probs, cutoff=cutoff
        )
        results[cutoff] = evaluate(references, predictions, betas)

    return results


def print_cutoff_sweep(results_by_cutoff):
    """
    Print the sweep and return the best cutoff by IoU.

    Args:
        results_by_cutoff (dict): output of `sweep_cutoff`.

    Returns:
        float: the cutoff with the highest IoU.
    """
    header = f"{'cutoff':<9}{'IoU':>8}{'rho':>8}{'P':>8}{'R':>8}{'F1':>8}"
    print(header)
    print("-" * len(header))

    for cutoff in sorted(results_by_cutoff):
        s = results_by_cutoff[cutoff]
        print(
            f"{cutoff:<9.2f}{s['iou']:>8.3f}{s['cor']:>8.3f}"
            f"{s['precision']:>8.3f}{s['recall']:>8.3f}{s['f1']:>8.3f}"
        )

    best = max(results_by_cutoff, key=lambda c: results_by_cutoff[c]["iou"])

    print("-" * len(header))
    print(f"best IoU at cutoff {best:.2f} ({results_by_cutoff[best]['iou']:.3f})")

    if best == min(results_by_cutoff):
        print(
            "NOTE: the best value is the lowest tested - the curve never "
            "turned over, so test below this before concluding."
        )

    return best


class ModelEvaluation:
    """
    Score a trained detector against gold validation and test.

    Args:
        config (ModelEvaluationConfig): paths and evaluation settings.
        predictor (Predictor): a ready inference wrapper.

    Usage::

        metrics = ModelEvaluation(config, predictor).run(val_refs, test_refs)
    """

    def __init__(self, config: ModelEvaluationConfig, predictor):
        self.config = config
        self.predictor = predictor

    def run(self, val_refs, test_refs):
        """
        Score baselines, the model, the threshold sweep and the zero-shot split.

        Args:
            val_refs (list[dict]): gold validation records.
            test_refs (list[dict]): gold test records.

        Returns:
            dict: every score computed, ready to serialise.
        """
        metrics = {}

        empty_val = sum(1 for r in val_refs if not r["hard_labels"])
        empty_test = sum(1 for r in test_refs if not r["hard_labels"])

        logger.info(
            f"validation: {len(val_refs)} items ({empty_val} with no gold "
            f"hallucination) | test: {len(test_refs)} items ({empty_test})"
        )

        # Baselines first: without them a system score is a number with no scale.
        for name, refs in (("validation", val_refs), ("test", test_refs)):
            metrics[f"{name}_mark_none"] = evaluate(
                refs, baseline_mark_none(refs), self.config.betas
            )
            metrics[f"{name}_mark_all"] = evaluate(
                refs, baseline_mark_all(refs), self.config.betas
            )
            print_scores(f"{name} | mark-none", metrics[f"{name}_mark_none"])
            print_scores(f"{name} | mark-all", metrics[f"{name}_mark_all"])

        # --- validation -----------------------------------------------------
        val_probs = self.predictor.predict_char_probs(val_refs)
        val_preds = build_prediction_records(val_refs, val_probs, OFFICIAL_CUTOFF)

        metrics["validation"] = evaluate(val_refs, val_preds, self.config.betas)
        metrics["validation_by_language"] = evaluate_by_language(
            val_refs, val_preds, self.config.betas
        )

        print()
        print_scores("validation | model", metrics["validation"])
        print()
        print("per language (validation)")
        print_language_table(metrics["validation_by_language"])

        # --- threshold sweep, validation only -------------------------------
        print()
        print("threshold sweep (validation only - never tuned on test)")
        sweep = sweep_cutoff(
            val_refs, val_probs, self.config.cutoff_sweep, self.config.betas
        )
        best_cutoff = print_cutoff_sweep(sweep)

        metrics["validation_cutoff_sweep"] = {
            str(cutoff): scores for cutoff, scores in sweep.items()
        }
        metrics["best_cutoff_on_validation"] = best_cutoff

        # --- test -----------------------------------------------------------
        test_probs = self.predictor.predict_char_probs(test_refs)
        test_preds = build_prediction_records(
            test_refs, test_probs, OFFICIAL_CUTOFF
        )

        metrics["test"] = evaluate(test_refs, test_preds, self.config.betas)
        metrics["test_by_language"] = evaluate_by_language(
            test_refs, test_preds, self.config.betas
        )

        print()
        print_scores("test | model", metrics["test"])
        print()
        print("per language (test)")
        print_language_table(metrics["test_by_language"])

        # --- zero-shot split ------------------------------------------------
        preds_by_id = {p["id"]: p for p in test_preds}

        surprise = [r for r in test_refs if r["lang"] in SURPRISE_LANGUAGES]
        seen = [r for r in test_refs if r["lang"] not in SURPRISE_LANGUAGES]

        metrics["test_seen_languages"] = evaluate(
            seen, [preds_by_id[r["id"]] for r in seen], self.config.betas
        )
        metrics["test_surprise_languages"] = evaluate(
            surprise, [preds_by_id[r["id"]] for r in surprise], self.config.betas
        )

        print()
        print_scores("test | languages seen in training", metrics["test_seen_languages"])
        print_scores("test | surprise languages (zero-shot)", metrics["test_surprise_languages"])

        self.save(metrics)

        return metrics

    def save(self, metrics):
        """Write the metrics to `config.metrics_path` as JSON."""
        with open(self.config.metrics_path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2, ensure_ascii=False)

        logger.info(f"metrics written to {self.config.metrics_path}")
