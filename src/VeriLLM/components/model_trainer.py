"""
Fine-tune the token classifier.

Split policy (decided, do not revert - CLAUDE.md section 6.2):

| role       | source                     | labels          |
|------------|----------------------------|-----------------|
| train      | silver `train_unlabeled`   | LLM ensemble    |
| eval/epoch | gold `validation` (499)    | **human**       |
| test       | gold `test` (1,902)        | human, held out |

No `train_test_split`. Holding a slice out of the silver data would only
measure agreement with our own annotators; evaluating on gold validation means
checkpoint selection is driven by human labels.

`eval_loss` rises while f1 improves, which is why `metric_for_best_model` is
f1 and not loss - cross-entropy punishes confidence, f1 scores decisions.
Selecting on loss picks epoch 1; selecting on f1 picks epoch 2.
"""

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from VeriLLM import logger
from VeriLLM.constants import LABEL_IGNORE_INDEX
from VeriLLM.entity.config_entity import ModelTrainerConfig


class DataCollatorForSoftTokenClassification(DataCollatorForTokenClassification):
    """
    Token-classification collator that also pads the float `soft_targets` column.

    The parent pads `input_ids`/`attention_mask` via the tokenizer and `labels`
    with -100, but has no idea what to do with a ragged float column -
    `tokenizer.pad` leaves it untouched and the tensor conversion then fails on
    rows of different length. This pulls the column out, lets the parent do its
    job, and pads the floats to the same width.

    Padded positions get 0.0. They are never read: the loss masks on
    `labels == -100`, and every padded position is -100 by construction.
    """

    def __call__(self, features):

        soft_targets = [list(feature.pop("soft_targets")) for feature in features]

        batch = super().__call__(features)

        width = batch["labels"].shape[1]

        padding_side = getattr(self.tokenizer, "padding_side", "right")

        if padding_side == "right":
            padded = [row + [0.0] * (width - len(row)) for row in soft_targets]
        else:
            padded = [[0.0] * (width - len(row)) + row for row in soft_targets]

        batch["soft_targets"] = torch.tensor(padded, dtype=torch.float32)

        return batch


class SoftLabelTrainer(Trainer):
    """
    Trainer whose loss is soft cross entropy against the ensemble vote fraction.

    Standard token classification maximises log p(hard label). This maximises
    the log-likelihood of a *distribution*::

        loss = -[ p * log q1 + (1 - p) * log q0 ]

    where `p` is the fraction of annotators who marked the token and `q` is the
    model's softmax. At p in {0, 1} it reduces exactly to cross entropy, so
    nothing is lost on unanimous tokens; the difference is entirely on the
    contested ones - span boundaries - which is where the ensemble disagrees.

    This is also the only loss here that optimises what rho measures. The hard
    label loss is indifferent between a confident wrong answer and a hedged one
    as long as the argmax matches; this one is not.

    Note the resolution cost of the three-vote ensemble: `p` now takes only
    four values {0, 1/3, 2/3, 1}, against 23 distinct values under the
    six-variant run.

    `log_softmax` is taken in float32 even under fp16, because the loss is a
    difference of logs and is where mixed precision most easily underflows.

    Falls back to the parent implementation when `soft_targets` is absent, so
    `trainer.predict()` on a plain tokenized dataset still works.
    """

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):

        soft_targets = inputs.pop("soft_targets", None)

        if soft_targets is None:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        labels = inputs["labels"]

        outputs = model(**inputs)

        logits = outputs.logits

        log_probs = torch.log_softmax(logits.float(), dim=-1)

        # -100 marks context tokens, specials and padding
        mask = labels != LABEL_IGNORE_INDEX

        if mask.any():

            target = soft_targets[mask]

            per_token = -(
                target * log_probs[..., 1][mask]
                + (1.0 - target) * log_probs[..., 0][mask]
            )

            loss = per_token.mean()

        else:
            # keeps the graph connected rather than returning a bare constant
            loss = logits.sum() * 0.0

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """
    Token-level precision, recall and f1 on the positive class.

    Deliberately NOT the reported result: it counts tokens, while the official
    metrics count characters. It exists to drive checkpoint selection and to
    answer one question quickly - did the model collapse to the majority class?

    Args:
        eval_pred: `(logits, labels)` from the Trainer.

    Returns:
        dict: precision, recall and f1 for the hallucinated class.
    """
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    mask = labels != LABEL_IGNORE_INDEX

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels[mask],
        predictions[mask],
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    return {"precision": precision, "recall": recall, "f1": f1}


class ModelTrainer:
    """
    Fine-tune a token classifier on silver labels, selecting on gold validation.

    Args:
        config (ModelTrainerConfig): encoder, hyperparameters and output paths.
        tokenizer: the same fast tokenizer used for encoding.

    Usage::

        trainer = ModelTrainer(config, tokenizer).train(train_ds, eval_ds)
    """

    def __init__(self, config: ModelTrainerConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def build_training_arguments(self):
        """
        Build `TrainingArguments` from config.

        lr 2e-5 is the shared task's own XLM-R baseline recipe
        (arXiv:2504.11975, footnote 7), which keeps the run comparable to it.
        The epoch count is ours: epochs 3-5 strictly hurt.
        """
        return TrainingArguments(
            output_dir=str(self.config.checkpoint_dir),
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            num_train_epochs=self.config.num_train_epochs,
            weight_decay=self.config.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=True,
            fp16=self.config.fp16 and torch.cuda.is_available(),
            seed=self.config.seed,
            logging_steps=50,
            report_to="none",
        )

    def train(self, train_dataset, eval_dataset):
        """
        Run training and save the best checkpoint.

        Args:
            train_dataset: tokenized silver rows.
            eval_dataset: tokenized gold validation rows.

        Returns:
            transformers.Trainer: the fitted trainer, best weights loaded.
        """
        model = AutoModelForTokenClassification.from_pretrained(
            self.config.model_name,
            num_labels=2,
        )

        if self.config.use_soft_labels:
            data_collator = DataCollatorForSoftTokenClassification(
                tokenizer=self.tokenizer
            )
            trainer_class = SoftLabelTrainer
        else:
            data_collator = DataCollatorForTokenClassification(
                tokenizer=self.tokenizer
            )
            trainer_class = Trainer
            # the parent Trainer would choke on an unexpected float column
            train_dataset = train_dataset.remove_columns("soft_targets")
            eval_dataset = eval_dataset.remove_columns("soft_targets")

        logger.info(
            f"training {self.config.model_name} for "
            f"{self.config.num_train_epochs} epochs | loss: "
            f"{'soft cross-entropy (vote fraction)' if self.config.use_soft_labels else 'hard cross-entropy'}"
        )

        trainer = trainer_class(
            model=model,
            args=self.build_training_arguments(),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        trainer.save_model(str(self.config.final_model_dir))
        self.tokenizer.save_pretrained(str(self.config.final_model_dir))

        logger.info(f"best checkpoint saved to {self.config.final_model_dir}")

        return trainer
