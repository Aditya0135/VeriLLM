"""
Pair encoding and label alignment.

The diagnosis in CLAUDE.md section 6.3 is implemented here: the model sees the
question (and optionally the retrieved reference) as context, not just the
answer. The answer-only setup is unlearnable as posed - 74.3% of token
instances appear with both labels in gold, capping any text-only model at
76.13% accuracy.

Only the answer is ever labelled. Context is evidence to attend to, never
something the model is asked to classify.
"""

from VeriLLM import logger
from VeriLLM.constants import LABEL_IGNORE_INDEX
from VeriLLM.entity.config_entity import TokenizationConfig
from VeriLLM.utils.labels import spans_to_character_array, spans_to_soft_array


class Tokenizer:
    """
    Encode (context, answer) pairs and project character labels onto tokens.

    Args:
        config (TokenizationConfig): encoder name, max length, reference flag.
        tokenizer: a loaded fast tokenizer. Passed in rather than constructed
            so the trainer and the predictor provably share one instance.

    Usage::

        encoder = Tokenizer(config, tokenizer)
        row = encoder.tokenize_and_align_labels(val[0])
        assert set(row["labels"]) <= {-100, 0, 1}
    """

    def __init__(self, config: TokenizationConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def encode_pair(self, question, answer, reference_ans=""):
        """
        Tokenize a (context, answer) pair, protecting the answer from truncation.

        The context is the question, optionally followed by the retrieved
        reference. Question first: ATLANTIS (arXiv:2508.05179) ablated
        placement and found the question at the beginning works best.

        XLM-R takes at most **two** segments (`type_vocab_size == 1`), so the
        reference cannot be a third sequence - it is concatenated into the
        first. Passing a third positional argument to the tokenizer does not
        create one: it binds to `text_target`, the seq2seq label field, and the
        answer then never reaches the model at all while `sequence_ids() == 1`
        silently selects question tokens instead.

        `truncation="only_first"` trims the context and never the answer, which
        is what keeps character offsets valid - they index the answer.

        One failure mode: when the answer *alone* exceeds `max_length` there is
        nothing left to trim off the context and the fast tokenizer raises. On
        this data that is 54 of 3,327 silver rows (1.6%), 5 of 499 validation
        rows (1.0%) and 14 of 1,902 test rows (0.7%). `longest_first` then
        trims the answer instead - the same recall loss the answer-only setup
        already accepted. `tokenizers` raises a bare `Exception`, hence the
        broad catch.

        Args:
            question (str): `model_input`.
            answer (str): `model_output_text`; returned offsets index this.
            reference_ans (str): the retrieved reference, or `""` for none.
                The gold splits carry no reference, so the default makes them
                degrade cleanly to question-only.

        Returns:
            transformers.BatchEncoding: carries `offset_mapping`, and
            `sequence_ids()` returns 1 on exactly the answer tokens.
        """
        if self.config.use_reference and reference_ans:
            context = f"{question} {reference_ans}".strip()
        else:
            context = question

        try:
            return self.tokenizer(
                context,
                answer,
                truncation="only_first",
                max_length=self.config.max_length,
                return_offsets_mapping=True,
            )

        except Exception:
            return self.tokenizer(
                context,
                answer,
                truncation="longest_first",
                max_length=self.config.max_length,
                return_offsets_mapping=True,
            )

    def tokenize_and_align_labels(self, example):
        """
        Tokenize one row and project its character labels onto tokens.

        Context tokens - question and reference - plus separators and specials
        all get `LABEL_IGNORE_INDEX`, so the loss sees exactly the span the
        official metric scores.

        A token is labelled 1 if **any** of its characters is hallucinated.
        This is slightly more recall-friendly than the official baseline, which
        labels a token only when it falls entirely inside a gold span.

        Rows come out unpadded; the collator pads each batch to its own longest
        row and pads `labels` with `LABEL_IGNORE_INDEX`.

        Works on the silver train split and on gold validation/test alike: all
        three store `hard_labels` as official `[start, end]` character spans
        and carry `model_input`. `reference_ans` is looked up with a default,
        so the gold splits fall back to question-only.

        Args:
            example (dict): a row with `model_input`, `model_output_text`,
                `hard_labels` and `soft_labels` in official span format;
                `reference_ans` optional.

        Returns:
            dict: `input_ids`, `attention_mask`, `labels`, `soft_targets`.
        """
        text = example["model_output_text"]

        # hard_labels holds official spans, so expand them back to characters
        char_labels = spans_to_character_array(text, example["hard_labels"])

        # the graded version of the same thing: what fraction of the ensemble
        # marked each character
        char_probs = spans_to_soft_array(text, example.get("soft_labels", []))

        # `reference_answer` is what stage 02 writes; `reference_ans` is the
        # column name the earlier notebook datasets used. Gold validation/test
        # have neither, and fall back to question-only.
        reference = (
            example.get("reference_answer")
            or example.get("reference_ans")
            or ""
        )

        encoding = self.encode_pair(example["model_input"], text, reference)

        sequence_ids = encoding.sequence_ids()

        token_labels = []
        token_soft_targets = []

        for index, (start, end) in enumerate(encoding["offset_mapping"]):

            # sequence 1 is the answer; everything else is context or special
            if sequence_ids[index] != 1 or start == end:
                token_labels.append(LABEL_IGNORE_INDEX)
                token_soft_targets.append(0.0)
                continue

            token_labels.append(1 if char_labels[start:end].any() else 0)

            # mean vote fraction over the characters this token covers; a token
            # straddling a span boundary lands between 0 and 1, which is
            # exactly the sub-token signal the hard labels round away
            token_soft_targets.append(float(char_probs[start:end].mean()))

        encoding["labels"] = token_labels
        encoding["soft_targets"] = token_soft_targets

        encoding.pop("offset_mapping")

        return encoding

    def tokenize_dataset(self, dataset, description=""):
        """
        Map `tokenize_and_align_labels` over a split, dropping the text columns.

        Args:
            dataset (datasets.Dataset): rows to encode.
            description (str): name used in the log line.

        Returns:
            datasets.Dataset: with `input_ids`, `attention_mask`, `labels`,
            `soft_targets` and nothing else.
        """
        tokenized = dataset.map(
            self.tokenize_and_align_labels,
            remove_columns=dataset.column_names,
        )

        logger.info(f"tokenized {description or 'dataset'}: {len(tokenized)} rows")

        return tokenized
