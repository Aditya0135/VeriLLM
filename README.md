# VeriLLM

VeriLLM detects hallucinated spans in LLM answers — **SemEval-2025 Task 3 (Mu-SHROOM)**.

Given a question and an LLM's answer, VeriLLM returns the character spans of the answer that are not
supported by a reference. Multilingual, 14 languages.

> Under active development.

## Pipeline

```
question + answer
  │
  ├─ retrieve a reference for the question (Tavily, translated into the question's language)
  │
  ├─ normalise text (NFC + quote/space fixes)
  │
  ├─ LLM annotator ensemble marks unsupported spans
  │     3 Groq models × 2 of 6 prompt variants = 6 votes per answer
  │
  ├─ aggregate votes → per-character scores → soft + hard labels
  │
  └─ fine-tune XLM-R token classification on those labels ──► character spans
```

Two ways to use it:

- **Without fine-tuning** — stop after the ensemble. The aggregated votes are already spans, so the
  pipeline runs directly on any question/answer pair.
- **With fine-tuning** — use the aggregated labels to train the token classifier, which then predicts
  spans in one forward pass without any API calls.

Predictions are scored with the task's official IoU and Spearman-ρ metrics.

## Layout

```
research/                             # all working code
  annote.ipynb                        # annotation + label aggregation
  Collecting_spans.ipynb              # annotation runs
  train.ipynb                         # fine-tune the detector
  train_with_ref.ipynb                # same, with the reference in the input
  validation_trained_classifier.ipynb # trained on the gold validation split
  show_test_examples.py               # print examples with underlined spans
  annotations.jsonl, reference_map.json   # cached API results — do not regenerate
  *-final/                            # trained checkpoints
src/VeriLLM/                          # package scaffolding, not yet implemented
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Only the annotation stage needs keys, in `.env`:

```
API_Tavily=...
GROQ_API_KEY1=...      # through GROQ_API_KEY5
```

Training and evaluation need none — the annotations are cached.

## Usage

Run the notebooks in `research/`. To inspect predictions:

```bash
python research/show_test_examples.py --langs EN HI
```

## Reference

Task and dataset: [arXiv:2504.11975](https://arxiv.org/pdf/2504.11975) ·
[Helsinki-NLP/mu-shroom](https://huggingface.co/datasets/Helsinki-NLP/mu-shroom)

MIT — see [LICENSE](LICENSE).
