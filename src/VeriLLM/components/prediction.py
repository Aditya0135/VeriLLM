"""
Inference: run the classifier and project its output back onto characters.

Kept separate from `model_evaluation` because this is also the serving path -
`app.py` needs spans for an arbitrary (question, answer) pair without importing
a scorer or a gold split.

Training and inference share `Tokenizer.encode_pair`, so the two cannot drift
apart. Scoring an answer-only input against a pair-trained model silently costs
several IoU points, and that class of bug is why the encoder is passed in
rather than rebuilt here.
"""

import numpy as np
import torch

from VeriLLM import logger
from VeriLLM.constants import OFFICIAL_CUTOFF
from VeriLLM.utils.labels import char_probs_to_hard_labels, char_probs_to_soft_labels


def build_reference_records(split):
    """
    Convert a gold Mu-SHROOM split into records the scorer accepts.

    Reads `model_output_text` **raw**: no normalization of any kind. Every gold
    offset indexes the untouched string, so normalizing here would silently
    shift all of them (CLAUDE.md section 5.2).

    Args:
        split (datasets.Dataset): `validation` or `test`, both labelled on the
            Hub.

    Returns:
        list[dict]: records with `id`, `lang`, `question`, `reference_ans`,
        `text`, `text_len`, `hard_labels` and `soft_labels`. The question is
        carried so inference can rebuild exactly the pair the model was
        trained on.
    """
    records = []

    for row in split:

        text = row["model_output_text"]

        records.append({
            "id": row["id"],
            "lang": row["lang"],
            "question": row["model_input"],
            # gold validation/test carry no reference; "" means question-only
            "reference_ans": (
                row.get("reference_answer") or row.get("reference_ans") or ""
            ),
            "text": text,
            "text_len": len(text),
            "hard_labels": [
                [int(start), int(end)] for start, end in row["hard_labels"]
            ],
            "soft_labels": [
                {
                    "start": int(span["start"]),
                    "end": int(span["end"]),
                    "prob": float(span["prob"]),
                }
                for span in row["soft_labels"]
            ],
        })

    return records


class Predictor:
    """
    Produce per-character hallucination probabilities for a set of records.

    Args:
        model: a loaded `AutoModelForTokenClassification`.
        encoder (Tokenizer): the same pair encoder used in training.
        device (str): "cuda" or "cpu".
        batch_size (int): rows per forward pass.

    Usage::

        predictor = Predictor(model, encoder)
        probs = predictor.predict_char_probs(val_refs)
    """

    def __init__(self, model, encoder, device=None, batch_size=16):
        self.model = model
        self.encoder = encoder
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        self.model.to(self.device)
        self.model.eval()

    def predict_char_probs(self, references):
        """
        Run the classifier and project token probabilities onto characters.

        Only answer tokens are projected back: question, reference and special
        tokens have their offsets blanked to `(0, 0)` first, so they can never
        write into `char_probs`.

        Where several tokens cover the same character the maximum is taken,
        not the mean - a character is hallucinated if any token covering it
        says so.

        Args:
            references (list[dict]): records from `build_reference_records`,
                or any dicts with `question`, `text`, `text_len` and
                optionally `reference_ans`.

        Returns:
            list[numpy.ndarray]: one float array per record, length
            `text_len`.
        """
        all_char_probs = []

        for start_index in range(0, len(references), self.batch_size):

            chunk = references[start_index:start_index + self.batch_size]

            encodings = [
                self.encoder.encode_pair(
                    record["question"],
                    record["text"],
                    record.get("reference_ans", ""),
                )
                for record in chunk
            ]

            answer_offsets = []

            for encoding in encodings:

                sequence_ids = encoding.sequence_ids()

                answer_offsets.append([
                    offset if sequence_id == 1 else (0, 0)
                    for offset, sequence_id in zip(
                        encoding["offset_mapping"], sequence_ids
                    )
                ])

            batch = self.encoder.tokenizer.pad(
                [
                    {
                        "input_ids": encoding["input_ids"],
                        "attention_mask": encoding["attention_mask"],
                    }
                    for encoding in encodings
                ],
                return_tensors="pt",
            )

            padded_length = batch["input_ids"].shape[1]

            for offsets in answer_offsets:
                offsets.extend([(0, 0)] * (padded_length - len(offsets)))

            inputs = {
                key: value.to(self.device) for key, value in batch.items()
            }

            with torch.no_grad():
                logits = self.model(**inputs).logits

            # probability of the "hallucinated" class for every token
            token_probs = torch.softmax(logits, dim=-1)[:, :, 1].cpu().numpy()

            for row_index, record in enumerate(chunk):

                char_probs = np.zeros(record["text_len"], dtype=float)

                for (start, end), prob in zip(
                    answer_offsets[row_index], token_probs[row_index]
                ):

                    # context, specials and padding were all blanked to (0, 0)
                    if start == end:
                        continue

                    char_probs[start:end] = np.maximum(
                        char_probs[start:end], prob
                    )

                all_char_probs.append(char_probs)

        logger.info(f"predicted character probabilities for {len(references)} rows")

        return all_char_probs

    def predict_spans(self, question, answer, reference_ans="",
                      cutoff=OFFICIAL_CUTOFF):
        """
        Convenience path for a single pair - the serving entry point.

        Args:
            question (str): the question.
            answer (str): the answer to mark up.
            reference_ans (str): optional retrieved reference.
            cutoff (float): decision threshold.

        Returns:
            dict: `{"hard_labels": [[start, end], ...],
                    "soft_labels": [{start, end, prob}, ...]}`
        """
        record = {
            "question": question,
            "text": answer,
            "text_len": len(answer),
            "reference_ans": reference_ans,
        }

        char_probs = self.predict_char_probs([record])[0]

        return {
            "hard_labels": char_probs_to_hard_labels(char_probs, cutoff),
            "soft_labels": char_probs_to_soft_labels(char_probs),
        }


def build_prediction_records(references, all_char_probs, cutoff=OFFICIAL_CUTOFF):
    """
    Package per-character probabilities into scorer-format prediction records.

    Produces both label kinds from the same vector: `hard_labels` for IoU and
    F-beta, `soft_labels` for rho. Deriving them from one array is what keeps
    them consistent.

    Args:
        references (list[dict]): the records the probabilities belong to.
        all_char_probs (list[numpy.ndarray]): output of `predict_char_probs`.
        cutoff (float): decision threshold for the hard labels.

    Returns:
        list[dict]: `{"id", "hard_labels", "soft_labels"}` per record.
    """
    return [
        {
            "id": reference["id"],
            "hard_labels": char_probs_to_hard_labels(char_probs, cutoff),
            "soft_labels": char_probs_to_soft_labels(char_probs),
        }
        for reference, char_probs in zip(references, all_char_probs)
    ]
