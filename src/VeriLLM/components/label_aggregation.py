"""
Turn annotator votes into silver labels.

Each valid vote becomes a per-character 0/1 array; the arrays are averaged and
the mean is converted straight into the official Mu-SHROOM schema, so what
lands on disk has the same shape as the gold `validation` and `test` splits -
`[[start, end], ...]` and `[{start, end, prob}, ...]`, not per-character arrays
wearing the same field names (CLAUDE.md section 5.1).

Two decisions worth knowing about:

**Errored votes are excluded from the average, not counted as zeros.** An empty
span list caused by a 429 is not evidence that the text is clean; counting them
dragged 243 rows to all-zero labels in the six-variant run.

**The threshold is `>=`, not `>`.** The official gold rule is strictly `>`, but
gold has ~3 annotators, where `>0.5` means 2-of-3. This ensemble now also casts
3 votes, so `>=0.5` reproduces that same 2-of-3 majority. A row that lost a
vote to an API error has only 2, where `>=0.5` means 1-of-2 and a single model
becomes decisive - `n_valid_votes` is stored per row so those can be filtered
or down-weighted later.
"""

import numpy as np

from VeriLLM import logger
from VeriLLM.entity.config_entity import LabelAggregationConfig
from VeriLLM.utils.common import load_jsonl
from VeriLLM.utils.labels import labels_to_spans, spans_to_character_labels


def valid_votes(annotation):
    """
    Return only the votes of an annotation that actually reached a model.

    Args:
        annotation (dict): one annotation record.

    Returns:
        list[dict]: votes without an `error` field.

    Example:
        >>> valid_votes({"metadata": [{"spans": []}, {"spans": [], "error": "429"}]})
        [{'spans': []}]
    """
    return [vote for vote in annotation["metadata"] if "error" not in vote]


def filter_annotations(annotations, min_valid_votes):
    """
    Split annotations into those with enough surviving votes and those without.

    Args:
        annotations (list[dict]): records read from the jsonl file.
        min_valid_votes (int): floor on surviving votes.

    Returns:
        tuple[list, list]: (kept, dropped).
    """
    kept = []
    dropped = []

    for annotation in annotations:

        if len(valid_votes(annotation)) >= min_valid_votes:
            kept.append(annotation)
        else:
            dropped.append(annotation)

    return kept, dropped


class LabelAggregation:
    """
    Aggregate the annotator ensemble into official-format silver labels.

    Args:
        config (LabelAggregationConfig): paths, vote floor and threshold.

    Usage::

        silver = LabelAggregation(config).run(train)
        silver.save_to_disk(config.dataset_path)
    """

    def __init__(self, config: LabelAggregationConfig):
        self.config = config

    def load_annotations(self):
        """
        Read the annotations file and drop rows with too few valid votes.

        Returns:
            list[dict]: the kept annotation records.
        """
        annotations = load_jsonl(self.config.annotations_path)

        if not annotations:
            raise FileNotFoundError(
                f"no annotations at {self.config.annotations_path} - run "
                f"stage_03_annotation first"
            )

        kept, dropped = filter_annotations(
            annotations, self.config.min_valid_votes
        )

        vote_counts = [len(valid_votes(a)) for a in kept]

        logger.info(
            f"annotations: {len(annotations)} read, {len(kept)} kept, "
            f"{len(dropped)} dropped for fewer than "
            f"{self.config.min_valid_votes} valid votes"
        )

        if vote_counts:
            lost = sum(
                1 for a in kept
                if len(valid_votes(a)) < len(a["metadata"])
            )
            logger.info(
                f"valid votes per kept row: min={min(vote_counts)}, "
                f"max={max(vote_counts)}; {lost} rows lost a vote to an API error"
            )

        return kept

    def run(self, dataset):
        """
        Attach silver `soft_labels` and `hard_labels` to the matching rows.

        Args:
            dataset (datasets.Dataset): rows carrying `id` and
                `model_output_text`.

        Returns:
            datasets.Dataset: only the annotated rows, plus `soft_labels`,
            `hard_labels`, `reference_ans`, `n_valid_votes` and `n_votes`.
        """
        annotations = self.load_annotations()

        by_id = {a["sample_id"]: a for a in annotations}

        dataset = dataset.filter(lambda row: row["id"] in by_id)

        threshold = self.config.hard_label_threshold

        def add_labels(example):
            annotation = by_id[example["id"]]
            text = example["model_output_text"]

            votes = valid_votes(annotation)

            if not votes:
                raise ValueError(
                    f"sample {example['id']} has no valid votes - filter first"
                )

            matrix = np.stack([
                spans_to_character_labels(text, vote.get("spans", []))
                for vote in votes
            ])

            # denominator is the number of votes that survived, not the number cast
            soft_char_labels = matrix.mean(axis=0).tolist()

            official = labels_to_spans(text, soft_char_labels, threshold=threshold)

            example["soft_labels"] = official["soft_labels"]
            example["hard_labels"] = official["hard_labels"]
            example["n_valid_votes"] = len(votes)
            example["n_votes"] = len(annotation["metadata"])
            example["reference_ans"] = annotation.get("reference_answer", "")

            return example

        labelled = dataset.map(add_labels, load_from_cache_file=False)

        positive_rate = float(np.mean([
            sum(end - start for start, end in row["hard_labels"])
            / max(len(row["model_output_text"]), 1)
            for row in labelled
        ]))

        logger.info(
            f"aggregated {len(labelled)} rows at threshold {threshold} - "
            f"{100 * positive_rate:.1f}% of characters positive "
            f"(gold: 39.6% validation / 41.5% test)"
        )

        return labelled
