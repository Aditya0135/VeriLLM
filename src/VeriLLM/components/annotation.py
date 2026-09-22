"""
The LLM annotator loop.

Sends one strict prompt to each of three Groq models and records the spans
they return. Three votes per sample, down from six.

Everything here is built around the fact that quota does not come back
(CLAUDE.md section 0):

* the output file is **append-only** and never truncated;
* `load_completed_samples` makes the loop resumable, so a crash at row 2,000
  costs nothing;
* every attempt rotates to a new API key, and a full pass through all keys
  failing is the only thing that triggers a sleep.

The output path comes from config and defaults to
`data/annotations_strict.jsonl`. The six-variant file at
`data/annotations.jsonl` is never opened for writing.
"""

import ast
import time
from datetime import datetime

from VeriLLM import logger
from VeriLLM.entity.config_entity import AnnotationConfig
from VeriLLM.prompts.annotation_prompts import PROMPT_VERSION, build_annotation_prompt
from VeriLLM.utils.common import append_jsonl
from VeriLLM.utils.llm_clients import create_llm_models, load_api_keys


def parse_spans(output):
    """
    Convert one model response into a list of span strings.

    The prompt demands a bare Python list, so `ast.literal_eval` is enough and
    is safe - unlike `eval`, it cannot execute anything. Anything unparseable
    becomes an empty list, which the aggregation step then treats as "this
    model marked nothing", not as an error.

    Args:
        output (str): raw text of the model response.

    Returns:
        list[str]: the span strings, possibly empty.

    Example:
        >>> parse_spans('["silver", "2008"]')
        ['silver', '2008']
        >>> parse_spans("I could not find any hallucinations.")
        []
    """
    if not output:
        return []

    text = output.strip()

    # models occasionally wrap the list in a markdown fence despite the prompt
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("python"):
            text = text[len("python"):]
        text = text.strip()

    try:
        spans = ast.literal_eval(text)
    except Exception:
        return []

    if isinstance(spans, str):
        return [spans]

    if not isinstance(spans, (list, tuple)):
        return []

    cleaned = []

    for span in spans:

        # a model returning a tuple instead of a string
        if isinstance(span, (tuple, list)):
            span = span[0] if span else None

        if isinstance(span, str) and span.strip():
            cleaned.append(span)

    return cleaned


# Substrings that mark an error no amount of retrying or key-rotating can fix.
# Without this a retired model id burns one attempt per key on every row -
# 17 keys x 3,351 rows x 2 dead models is roughly 114,000 wasted calls.
_PERMANENT_ERROR_MARKERS = (
    "model_not_found",
    "does not exist or you do not have access",
    "decommissioned",
    "invalid_api_key",
    "context_length_exceeded",
)


def is_permanent_error(message):
    """
    Decide whether an API error is worth retrying.

    Rate limits, timeouts and 5xx are transient - a different key or a short
    wait fixes them. A retired model id or a malformed request is not: every
    key will fail identically, so the loop should give up at once.

    Args:
        message (str): the exception text.

    Returns:
        bool: True when retrying cannot help.

    Example:
        >>> is_permanent_error("Error code: 404 - model_not_found")
        True
        >>> is_permanent_error("Error code: 429 - rate_limit_exceeded")
        False
    """
    lowered = (message or "").lower()

    return any(marker in lowered for marker in _PERMANENT_ERROR_MARKERS)


class Annotator:
    """
    Run the strict-prompt annotator ensemble over a dataset.

    Args:
        config (AnnotationConfig): models, paths, retry policy.

    Usage::

        annotator = Annotator(config)
        annotator.run(train, limit=5)
    """

    def __init__(self, config: AnnotationConfig):
        self.config = config
        self.api_keys = load_api_keys()
        self.llm_models = create_llm_models(
            config.models,
            self.api_keys,
            temperature=config.temperature,
        )

    def _log(self, message):
        """Write one timestamped line to the annotation log and to the logger."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.config.log_path, "a", encoding="utf-8") as file:
            file.write(f"[{timestamp}] {message}\n")

        logger.info(message)

    def load_completed_samples(self):
        """
        Read the ids already present in the output file.

        Returns:
            set: sample ids to skip. Empty when the file does not exist yet.
        """
        import json
        import os

        completed = set()

        if not os.path.exists(self.config.annotations_path):
            return completed

        with open(self.config.annotations_path, "r", encoding="utf-8") as file:

            for line in file:

                try:
                    completed.add(json.loads(line)["sample_id"])
                except Exception:
                    continue

        return completed

    def annotate_sample(self, sample_id, sample):
        """
        Collect one vote from each model for a single sample.

        Every attempt takes a fresh key from that model's cycle, including the
        first. Only when a full pass through all keys has failed in a row is
        the failure treated as a genuine rate limit and slept off; otherwise
        the loop just moves to the next key.

        A vote that never succeeds is recorded with an `error` field rather
        than dropped, so aggregation can tell "marked nothing" apart from
        "never answered" - counting a 429 as an empty span list dragged 243
        rows to all-zero labels in the previous run.

        Args:
            sample_id: the row id.
            sample (dict): row with `model_input`, `model_output_text` and
                `reference_answer`.

        Returns:
            dict: one annotation record, ready to append.
        """
        prompt = build_annotation_prompt(
            sample["model_input"],
            sample["model_output_text"],
            sample.get("reference_answer", ""),
        )

        metadata = []

        # at least one full pass through every key before giving up
        max_retry = max(self.config.max_retries, len(self.api_keys))

        for model_name, key_cycle in self.llm_models.items():

            last_error = None

            for attempt in range(max_retry):

                key_id, llm = next(key_cycle)

                try:
                    response = llm.invoke(prompt)
                    spans = parse_spans(response.content)

                    metadata.append({
                        "model": model_name,
                        "prompt_version": PROMPT_VERSION,
                        "spans": spans,
                    })

                    self._log(
                        f"Sample {sample_id} | {model_name} | key {key_id} | "
                        f"SUCCESS | {len(spans)} spans"
                    )
                    break

                except Exception as error:
                    last_error = str(error)

                    self._log(
                        f"Sample {sample_id} | {model_name} | key {key_id} | "
                        f"FAILED | attempt {attempt + 1} | {last_error}"
                    )

                    if is_permanent_error(last_error):
                        # every key will fail identically - stop immediately
                        self._log(
                            f"Sample {sample_id} | {model_name} | PERMANENT "
                            f"error, not retrying"
                        )

                        metadata.append({
                            "model": model_name,
                            "prompt_version": PROMPT_VERSION,
                            "spans": [],
                            "error": last_error,
                        })
                        break

                    if attempt == max_retry - 1:
                        self._log(
                            f"Sample {sample_id} | {model_name} | giving up "
                            f"after {max_retry} attempts"
                        )

                        metadata.append({
                            "model": model_name,
                            "prompt_version": PROMPT_VERSION,
                            "spans": [],
                            "error": last_error,
                        })

                    elif (attempt + 1) % len(self.api_keys) == 0:
                        # every key failed in a row -> genuinely rate limited
                        self._log(
                            f"all {len(self.api_keys)} keys failed; sleeping "
                            f"{self.config.retry_wait_seconds}s"
                        )
                        time.sleep(self.config.retry_wait_seconds)

                    else:
                        time.sleep(1)

            # stay safely below Groq RPM limits
            time.sleep(self.config.rpm_sleep_seconds)

        return {
            "sample_id": sample_id,
            "prompt_version": PROMPT_VERSION,
            "model_input": sample["model_input"],
            "model_output_text": sample["model_output_text"],
            "reference_answer": sample.get("reference_answer", ""),
            "metadata": metadata,
        }

    def run(self, dataset, limit=None):
        """
        Annotate every row not already in the output file.

        Args:
            dataset (datasets.Dataset): rows with `id`, `model_input`,
                `model_output_text` and `reference_answer`.
            limit (int | None): stop after this many *newly annotated* rows.
                Used for the smoke test; `None` processes everything.

        Returns:
            int: how many rows were newly annotated.
        """
        completed = self.load_completed_samples()

        self._log(
            f"starting annotation | prompt {PROMPT_VERSION} | "
            f"{len(self.llm_models)} models | {len(completed)} rows already done | "
            f"output {self.config.annotations_path}"
        )

        written = 0

        for sample in dataset:

            if limit is not None and written >= limit:
                break

            sample_id = sample["id"]

            if sample_id in completed:
                continue

            record = self.annotate_sample(sample_id, sample)

            append_jsonl(self.config.annotations_path, record)

            written += 1

            self._log(f"Saved sample {sample_id}  ({written} this run)")

        self._log(f"annotation run finished: {written} new rows")

        return written
