"""
The annotator prompt.

One strict prompt, sent to all three models. This replaces the six-variant
ensemble (2 sampled per sample = 6 votes) that produced
`data/annotations.jsonl`; that file is kept untouched as the control arm of a
prompt ablation.

Why one strict prompt
---------------------
Measured on the 46 rows that were annotated with all six variants, the
variants differ far less than expected - 65.3% of characters marked by the
"smallest span" variant against 74.9% by the default, against a gold rate of
39.6% (validation) / 41.5% (test). Corpus-wide the six-variant ensemble
aggregates to 47.1%, so the pipeline over-marks by roughly 6 points.

The wording below therefore does two things none of the six variants did:

1. It uses the **official annotator guideline verbatim** - "the minimum number
   of characters that would need to be edited or deleted" (arXiv:2504.11975).
   The gold labels were produced by humans following that instruction, so
   matching it is the closest available alignment with the target.
2. It gives a **worked micro-example** of word-level granularity. Abstract
   instructions to be brief bought only ~10 points across the six variants; a
   demonstration is a stronger signal than an adjective.

Cost of the change: with 1 prompt x 3 models the soft labels take only four
values {0, 1/3, 2/3, 1}, against 23 distinct values before. `SoftLabelTrainer`
is the only loss that optimises rho, so this trades resolution on the
project's strongest metric for precision on its weakest. That trade is
deliberate.

Bump `PROMPT_VERSION` on any edit: it is written into every annotation record,
so a mixed-prompt file stays diagnosable.
"""

PROMPT_VERSION = "strict-minimal-v1"


SYSTEM_MESSAGE = (
    "You are a hallucination detection assistant. You mark the minimum "
    "number of characters that would have to be edited or deleted to make "
    "the LLM answer correct."
)


USER_TEMPLATE = """Compare the LLM answer with the reference answer.

Return ONLY a Python list of strings:
["hallucinated_span_1", "hallucinated_span_2"]

Rules:
- Copy spans EXACTLY from the LLM answer. Never paraphrase or correct.
- Mark the MINIMUM characters that would need to be edited or deleted to make
  the answer correct. Be conservative.
- Prefer content words. Do not include articles, prepositions, conjunctions or
  punctuation unless they are themselves wrong.
- Mark the wrong word or number itself, not the sentence containing it.
  Example: in "won a silver medal in 2008", if only the metal is wrong,
  return ["silver"] - not the whole phrase.
- Do not mark information that is merely absent from the reference.
- Do not mark correct paraphrases or equivalent wording.
- If nothing is hallucinated, return [].
- Do not provide explanations.

Question:
{model_input}

LLM Answer:
{model_output_text}

Reference Answer:
{reference_answer}"""


def build_annotation_prompt(model_input, model_output_text, reference_answer):
    """
    Build the single annotation prompt for one sample.

    Args:
        model_input (str): the question.
        model_output_text (str): the answer whose spans are to be marked.
        reference_answer (str): the retrieved reference to check against.

    Returns:
        list[tuple[str, str]]: `[(role, content), ...]` in the shape
        `ChatGroq.invoke` accepts.

    Example:
        >>> messages = build_annotation_prompt("Who won?", "Bob won.", "Alice won.")
        >>> len(messages), messages[0][0]
        (2, 'system')
    """
    return [
        ("system", SYSTEM_MESSAGE),
        (
            "user",
            USER_TEMPLATE.format(
                model_input=model_input or "",
                model_output_text=model_output_text or "",
                reference_answer=reference_answer or "",
            ),
        ),
    ]
