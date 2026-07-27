# VeriLLM — Mu-SHROOM Hallucination Span Detection

Project context file. Read this first; it replaces re-reading the paper/dataset every session.

---

## 0. Scope and constraints — read before proposing work

- **University coursework project.** Not a competition entry. The SemEval evaluation window is closed; there is no leaderboard to submit to and no submission-format requirement. The paper's numbers are a **reference point for the report**, not a target to chase.
- **Everything so far is exploratory.** Partial runs are expected and fine — 268/3350 annotated samples is a working state, not a defect.
- **API quota is the binding constraint.** Free-tier Groq (5 rotated keys) and Tavily. Every LLM/search call costs quota that doesn't come back, which is why results are cached to disk (`reference_map.json`, `annotations.jsonl`) and every loop is resumable. **Treat these caches as the expensive artifact of the project.** Never regenerate, overwrite, or clear them to "start fresh"; always skip work that's already cached.
- **Therefore, prefer offline work.** Anything that can be done by re-reading `annotations.jsonl`, re-aggregating labels, changing thresholds, retraining, or scoring against gold costs nothing and can be iterated freely. Anything that issues new API calls should be scoped to the smallest useful sample.

---

## 1. What the task is

**SemEval-2025 Task 3: Mu-SHROOM** ([arXiv:2504.11975](https://arxiv.org/pdf/2504.11975), [HF: Helsinki-NLP/mu-shroom](https://huggingface.co/datasets/Helsinki-NLP/mu-shroom))

Given a **question** and an **LLM's answer**, mark the **character spans in the answer that are hallucinated** — i.e. content whose facts are not supported by a reference (Wikipedia page the question was written from).

It is a **character-level span-labeling** problem, not sentence/document classification. Multilingual: 14 languages.

Official definition used by the annotators — worth matching, because the gold labels follow it:
> Highlight the **minimum number of characters** that would need to be edited or deleted to make the answer correct. Be conservative; prefer content words over function words.

---

## 2. Dataset rules (the part that matters)

### Splits

| Split | Rows (`all` config) | Labels? | Languages |
|---|---|---|---|
| `train_unlabeled` | 3,350 | **No** (`hard_labels`/`soft_labels` are `None`) | EN, ES, FR, ZH only |
| `validation` | 499 | Yes (~50/lang) | AR, DE, EN, ES, FI, FR, HI, IT, SV, ZH |
| `test` | 1,904 | **Yes** — released post-competition (~150/lang) | above 10 + surprise: CA, CS, EU, FA |

- Per-language configs exist too: `load_dataset("Helsinki-NLP/mu-shroom", "en")`.
- **`train_unlabeled` rows have `id = None`** → must assign your own index (`add_id` with `with_indices=True`). Also no `wikipedia_url`.
- **The test split is labeled on HF**, so you can self-score offline with zero API cost. This is what the report's numbers should come from — not from our own silver-labeled train split, which only measures agreement with the LLM annotators.
- Test-only "surprise" languages (CA, CS, EU, FA) have no train/val → any system must be zero-shot multilingual for these.

### Record fields

```python
{
  "id":                  "val-en-1",            # None in train_unlabeled
  "lang":                "EN",
  "model_input":         "What did Petra van Staveren win a gold medal for?",
  "model_output_text":   "Petra van Stoveren won a silver medal in the 2008 Summer Olympics in Beijing, China.",
  "model_id":            "tiiuae/falcon-7b-instruct",
  "wikipedia_url":       "https://en.wikipedia.org/wiki/Petra_van_Staveren",
  "hard_labels":         [[25, 31], [45, 49], [69, 83]],
  "soft_labels":         [{"start": 25, "end": 31, "prob": 0.9}, ...],
  "model_output_logits": [...],                 # per token of the generating model
  "model_output_tokens": ["Pet","ra","Ġvan",...],
  "annotations":         [{"annotator_id": "...", "labels": [[25,32],...]}, ...],  # raw per-annotator
}
```

### Label format — **critical**

- `hard_labels`: list of `[start, end)` **character** offsets into `model_output_text`. End-exclusive. Empty list `[]` = no hallucination (this happens; see metric special case).
- `soft_labels`: list of contiguous `{start, end, prob}` segments tiling the text, where `prob` = fraction of annotators who marked those characters. Derived from `annotations`.
- Offsets index **characters of `model_output_text`**, not tokens. `model_output_tokens`/`logits` come from the *generating* model and do not align to your tokenizer.
- Hard labels are the **>50% majority binarization** of soft labels (official rule; we currently use 0.7 for our silver labels — see §5).

### Evaluation metrics (official)

Per datapoint, over character index sets:

1. **IoU** — `|Ĉ ∩ C| / |Ĉ ∪ C|`, where `C` = gold characters with annotator prob > 0.5, `Ĉ` = predicted characters. **Primary ranking metric.**
2. **ρ** — Spearman correlation between the per-character gold probability vector and the per-character predicted probability vector. Tie-breaker; measures calibration to human disagreement.

Special case for items with **no** gold hallucination: IoU = 1 if your prediction is also empty else 0; ρ = 1 if you assign the same probability to every character else 0.

Default conversions if you only produce one: continuous → binary at cutoff 0.5; binary → continuous as 1.0/0.0.

**Do not report token-level precision/recall/F1 as the result.** Convert token predictions → character spans → IoU/ρ.

### Reference scores to beat

| | IoU |
|---|---|
| `mark-none` baseline | very low (data is biased toward hallucinated items) |
| `mark-all` baseline | modest, but far below top teams |
| XLM-R token-classification baseline | ranks extremely low |
| Mean over all 43 teams, per language | 0.31 (ES) – 0.51 (IT) |
| Best system per language | 0.53 (ES) – 0.78 (IT) |
| Human inter-annotator agreement | 0.45 (EN) – 0.87 (IT) |

Findings from the paper that should steer design decisions:
- **RAG helps most** — 52.6% of submissions used it, and they scored significantly higher on both metrics. The core difficulty is *finding the right reference*, not the classifier.
- Prompt-only systems had significantly **lower ρ**; fine-tuning-based approaches were stronger.
- Teams that relied **mainly on the provided data** scored significantly lower — synthetic/external data matters.
- Base-model families that underperformed: **BERT-family, Llama, Flan-T5**. Outperformed: DeepSeek, Qwen, Claude, GPT.
- Annotator agreement correlates only weakly with system score — low-agreement items are hard for other reasons too.

---

## 3. Our approach (VeriLLM)

Because `train_unlabeled` has no labels, we **generate silver labels with an LLM annotator ensemble**, then fine-tune a token classifier on them.

```
train_unlabeled (3350, no labels)
  │
  ├─ Tavily web search per unique question ──► reference_answer          [reference_map.json]
  │     └─ langdetect + Groq llama-3.1-8b-instant translate ref into the question's language
  │
  ├─ Unicode normalization (NFC + quote/space fixes) on input/output/reference
  │
  ├─ LLM-as-annotator ensemble: 3 Groq models × 2 prompts (sampled from 6 variants)
  │     = 6 votes/sample, each returning exact-copy hallucinated spans   [annotations.jsonl]
  │
  ├─ spans → per-character 0/1 vector per vote; stack → soft = mean, hard = (soft >= 0.7)
  │                                                    [temp_mu_shroom_llm_annotated_train/]
  │
  └─ mBERT token classification (offset-mapping alignment) ──► ./mbert_hallucination_detector
```

**Annotator ensemble** (`create_llm_models`, Groq): `qwen/qwen3.6-27b`, `llama-3.3-70b-versatile`, `openai/gpt-oss-120b`, all `temperature=0`.

**6 prompt variants** (2 randomly sampled per sample) differ in span-granularity policy: default / atomic-claim decomposition / strict-contradiction-only / complete-claim-with-context / smallest-span / context-aware-balanced. Deliberate diversity — it produces the disagreement that makes soft labels meaningful.

All prompts require: spans **copied exactly** from the answer, no paraphrase, `[]` if none, no explanations. Parsed with `ast.literal_eval`.

**Robustness already built in**: `annotations.jsonl` is append-only and resumable via `load_completed_samples()`; `reference_map.json` is written after every search; 5 retries with 60s backoff; 2.2s sleep for Groq RPM; everything logged to `annotation.log`.

---

## 4. File map

| Path | What it is |
|---|---|
| `research/BERT_classifier_final copy.ipynb` | **Most complete pipeline.** Annotation → aggregation → char labels → mBERT training → eval. Currently ahead of the non-copy version. Untracked in git. |
| `research/BERT_classifier_final.ipynb` | Clean annotation-generation pipeline (steps 1–7), stops after the annotation loop. |
| `research/BERT_classifier.ipynb` | Original scratch/experiment notebook. Contains a **hardcoded Tavily API key** — remove before any commit/push. |
| `research/reference_map.json` | `{question: reference_answer}` cache from Tavily. Skips already-searched questions. |
| `research/annotations.jsonl` | One line per sample: `{sample_id, model_input, model_output_text, reference_answer, metadata:[{model, prompt, spans}]}`. **268 of 3350 samples done.** |
| `research/annotation.log` | Timestamped per-call success/failure log. |
| `research/temp_mu_shroom_llm_annotated_train/` | `save_to_disk` HF dataset with silver `soft_labels`/`hard_labels`. |
| `research/mbert-hallucination-detector/` | Trainer checkpoints (24/48/72/96/120). Best = **checkpoint-48**, eval_loss 0.4334. |
| `research/mbert_hallucination_detector/` | Final saved model + tokenizer. |
| `src/VeriLLM/`, `app.py`, `main.py`, `config/`, `params.yaml` | Empty scaffolding from `template.py`. Nothing ported out of the notebooks yet. |
| `.env` | `API_Tavily`, `GROQ_API_KEY1`…`GROQ_API_KEY5` (rotated across models to spread rate limits). |

**Training config** (`bert-base-multilingual-uncased`, `num_labels=2`): lr 2e-5, batch 8, 5 epochs, weight_decay 0.01, fp16, eval+save per epoch, `load_best_model_at_end`, 80/20 split of the silver data. 120 steps total ⇒ ~215 training samples — this is a smoke-test-scale run, not a real result.

---

## 5. Invariants and known gotchas

1. **Two different label representations are in play.** The official format is a *list of `[start, end]` spans*. Our pipeline stores `hard_labels`/`soft_labels` as *per-character arrays of length `len(text)`*. Fine internally, but **convert with `labels_to_spans()` before scoring against gold**. Never feed a per-character array to something expecting official format, or vice versa.
2. **Never run `normalize_text` on validation/test.** It changes string length and therefore invalidates the gold character offsets. It is only safe on `train_unlabeled`, where labels are derived *after* normalization.
3. **`spans_to_character_labels` uses `text.find(span)`** — it labels only the *first* occurrence and silently drops any span the LLM didn't copy verbatim. Both are silent label loss. Worth measuring the unmatched-span rate and adding all-occurrence + fuzzy fallback matching.
4. **Threshold mismatch:** we binarize silver labels at `soft >= 0.7`; the official gold binarization is `> 0.5`. Intentional (precision over recall on noisy LLM votes), but keep it a named constant and ablate it.
5. **Soft labels are currently computed but unused in training.** The model trains on hard labels with cross-entropy, so it has nothing optimizing ρ. To score ρ: take `softmax(logits)[:, 1]` per token and broadcast to that token's character range via offset mapping. Soft-label (distillation / soft cross-entropy) training is the natural next step.
6. **Evaluation gap:** the current eval loop scores token-level predictions on a split of our own silver data. That measures agreement with the LLM annotators, not with humans. Real evaluation = official IoU + ρ against `validation` / `test` gold.
7. **Empty-prediction handling.** Items with no hallucination need the special-case rule (§2) or the IoU averages will be wrong.
8. **mBERT is a weak base per the paper** (BERT-family underperformed; XLM-R was the baseline). `xlm-roberta-base`/`large` is a cheap, likely-positive swap since the tokenizer/offset code is identical.
9. **Tavily's `include_answer="basic"` gives a short snippet, not the Wikipedia page.** The dataset's own reference is `wikipedia_url`. Given the paper's finding that retrieval quality dominates, fetching the actual Wikipedia article (available for val/test) is likely the highest-leverage improvement available.
10. **Secrets:** keys live in `.env`. `research/BERT_classifier.ipynb` cell 15 has a Tavily key in plaintext — strip it before committing.

---

## 6. Status

**Done:** dataset loading, reference retrieval + translation + caching, normalization, 6-prompt × 3-model annotation ensemble with resumability, span→char-label aggregation, mBERT token-classification training loop, token-level eval loop.

**Open work, ordered by value-per-API-call. Everything in the first group is free.**

*No API cost — do these first:*
1. Implement the **official scorer** (char-level IoU + ρ, with empty-item special cases). Without it there is no honest number for the report.
2. Run `mark-all` / `mark-none` baselines on `validation` — cheap, and they make any improvement legible.
3. Score the current mBERT checkpoint on `validation`/`test`. Even a weak number beats no number.
4. Use soft labels in the loss and emit continuous per-character probabilities, so ρ is measurable at all.
5. Swap mBERT → XLM-R and compare. Same offset-mapping code, and the paper suggests BERT-family is the weaker base.
6. Measure the silent-label-loss rate from `text.find` (§5.3) and the effect of the 0.7 vs 0.5 threshold (§5.4) — both are pure re-aggregation over existing `annotations.jsonl`.
7. Port the notebook pipeline into `src/VeriLLM/`.

*Costs API quota — scope deliberately:*
8. Extend annotation beyond 268 samples. More useful per call if spent on a **balanced sample across EN/ES/FR/ZH** than on continuing sequentially, since the current 268 are whatever came first in the split.
9. Retrieve real Wikipedia articles instead of Tavily snippets. Highest-leverage change per the paper, but re-annotation with better references means re-spending annotation quota — so validate it on a small subset before committing.

**Good report material regardless of scores:** per-language breakdown, the LLM-annotator agreement analysis you can already compute from the 6 votes in `annotations.jsonl` (compare it to the paper's human IAR of 0.45–0.87), threshold/base-model ablations, and honest error analysis of the failure modes in §5.
