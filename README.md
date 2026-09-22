# VeriLLM

VeriLLM detects hallucinated spans in LLM answers — **SemEval-2025 Task 3 (Mu-SHROOM)**.

Given a question and an LLM's answer, VeriLLM returns the character spans of the answer that are not
supported by a reference. Multilingual, 14 languages.

> Under active development.

## Pipeline

```
question + answer
  │
  ├─ retrieve a reference for the question (cached; Tavily only for new ones)
  │
  ├─ normalise text (NFC + quote/space fixes)   — training split only
  │
  ├─ LLM annotator ensemble marks unsupported spans
  │     3 Groq models × 1 strict prompt = 3 votes per answer
  │
  ├─ aggregate votes → per-character scores → soft + hard labels
  │
  └─ fine-tune XLM-R token classification on those labels ──► character spans
```

The annotator prompt asks for **the minimum characters that would have to be
edited or deleted** to make the answer correct — the same instruction the gold
annotators were given. It replaces an earlier six-variant ensemble that marked
47% of all characters against gold's ~40%.

Two ways to use it:

- **Without fine-tuning** — stop after the ensemble. The aggregated votes are already spans, so the
  pipeline runs directly on any question/answer pair.
- **With fine-tuning** — use the aggregated labels to train the token classifier, which then predicts
  spans in one forward pass without any API calls.

Predictions are scored with the task's official IoU and Spearman-ρ metrics.

## Layout

```
src/VeriLLM/
  constants/        OFFICIAL_CUTOFF, LABEL_IGNORE_INDEX, config paths
  entity/           one frozen dataclass per stage
  config/           ConfigurationManager — the only reader of the two YAMLs
  prompts/          the annotator prompt, versioned
  utils/
    common.py       yaml / json / jsonl helpers (jsonl append is never truncating)
    labels.py       the single owner of span <-> per-character conversion
    llm_clients.py  Groq clients + API-key rotation
  components/       data_ingestion, text_normalization, reference_retrieval,
                    annotation, label_aggregation, tokenization, model_trainer,
                    prediction, scorer, model_evaluation, plots
  pipeline/
    stage_01..06    individual stages, each runnable with -m
    silver_pipeline      stages 01-04 — every API call lives here
    supervised_pipeline  stages 05-06 — zero API calls, re-run freely
config/config.yaml  paths
params.yaml         hyperparameters
data/               cached API results — never regenerate
research/           the original notebooks this was ported from
```

The `silver` / `supervised` split is the project's main safety boundary:
nothing in `supervised_pipeline` can reach an API or touch a cache, so it can
be re-run without limit.

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

```bash
python main.py                  # aggregate cached votes, train, evaluate
python main.py --eval-only      # score the saved checkpoint only
python main.py --annotate       # ... including a fresh annotation pass
```

Annotation is off by default — it is the only stage that spends Groq quota
(3 models × 3,351 rows ≈ 10,000 calls, several hours). It is append-only and
resumable, so an interrupted run resumes exactly where it stopped. Smoke-test
it first:

```bash
python -m VeriLLM.pipeline.stage_03_annotation --limit 5
```

Individual stages run the same way:

```bash
python -m VeriLLM.pipeline.stage_04_label_aggregation
python -m VeriLLM.pipeline.supervised_pipeline --eval-only
```

To inspect predictions:

```bash
python research/show_test_examples.py --langs EN HI
```

## Reference

Task and dataset: [arXiv:2504.11975](https://arxiv.org/pdf/2504.11975) ·
[Helsinki-NLP/mu-shroom](https://huggingface.co/datasets/Helsinki-NLP/mu-shroom)

MIT — see [LICENSE](LICENSE).
