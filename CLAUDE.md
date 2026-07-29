# VeriLLM — Mu-SHROOM Hallucination Span Detection

Project context file. Read this first; it replaces re-reading the paper/dataset every session.

---

## 0. Scope and constraints — read before proposing work

- **University coursework project.** Not a competition entry. The SemEval evaluation window is closed; there is no leaderboard to submit to and no submission-format requirement. The paper's numbers are a **reference point for the report**, not a target to chase.
- **Annotation is COMPLETE.** All 3,351 `train_unlabeled` rows are annotated in `annotations.jsonl` (20,481 votes, 802 errored). Do not re-run the annotation loop; it costs quota and produces nothing new.
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
| `train_unlabeled` | **3,351** | **No** (`hard_labels`/`soft_labels` are `None`) | FR 1850, EN 809, ES 492, zh 200 |
| `validation` | 499 | Yes (~50/lang) | AR, DE, EN, ES, FI, FR, HI, IT, SV, ZH |
| `test` | **1,902** | **Yes** — released post-competition | 150/lang for the 10 above + ~100 each for surprise CA, CS, EU, FA |

Verified counts (not the paper's rounded ones). Note `test` is **unbalanced**: 150 for main languages, ~100 for the four surprise ones — so a pooled mean under-weights exactly the hardest rows. **Report the macro-average over languages**, which is what the paper's Table 4 is comparable to.

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
| `research/annote.ipynb` | **Annotation + aggregation pipeline** (was `model_training.ipynb`, renamed). Builds `temp_mu_shroom_llm_annotated_train_new` from `annotations.jsonl`. Change the threshold here. |
| `research/train.ipynb` | **The live training + evaluation notebook.** Loads the saved silver dataset, tokenizes, fine-tunes mBERT, then runs the full official Mu-SHROOM scorer. This is where work continues. |
| `research/Collecting_spans.ipynb` | Annotation collection runs. |
| `research/validation_trained_classifier.ipynb` | Separate experiment training on the *validation* split (produced `val-trained-detector/`, `val_trained_detector/`). Not part of the main line. |
| `research/BERT_classifier.ipynb` | Original scratch notebook. Contains a **hardcoded Tavily API key** — remove before any commit/push. |
| `research/reference_map.json` | `{question: reference_answer}` cache from Tavily. |
| `research/annotations.jsonl` | One line per sample: `{sample_id, model_input, model_output_text, reference_answer, metadata:[{model, prompt, spans}]}`. **COMPLETE: 3,351/3,351.** Never regenerate. |
| `research/annotation.log` | Timestamped per-call success/failure log. |
| `research/temp_mu_shroom_llm_annotated_train_new/` | `save_to_disk` HF dataset, **3,327 rows**. `hard_labels`/`soft_labels` stored in **official format** (span pairs / `{start,end,prob}`), same schema as gold val/test. Currently built at threshold **0.5** (40.15% positive chars). |
| `research/mbert-hallucination-detector-new/` | Trainer checkpoints. Best = **checkpoint-822** (epoch 2), f1 0.5000. |
| `research/mbert_hallucination_detector_new/` | Final saved model + tokenizer. |
| `src/VeriLLM/`, `app.py`, `main.py`, `config/`, `params.yaml` | Empty scaffolding from `template.py`. Nothing ported out of the notebooks yet. |
| `.env` | `API_Tavily`, `GROQ_API_KEY1`…`GROQ_API_KEY5` (rotated across models to spread rate limits). |

**Training config** (`bert-base-multilingual-uncased`, `num_labels=2`): lr 2e-5, batch 8, 5 epochs, weight_decay 0.01, fp16, eval+save per epoch, `load_best_model_at_end`, 80/20 split of the silver data. 120 steps total ⇒ ~215 training samples — this is a smoke-test-scale run, not a real result.

---

## 5. Invariants and known gotchas

1. **Two different label representations are in play.** The official format is a *list of `[start, end]` spans*. Our pipeline stores `hard_labels`/`soft_labels` as *per-character arrays of length `len(text)`*. Fine internally, but **convert with `labels_to_spans()` before scoring against gold**. Never feed a per-character array to something expecting official format, or vice versa.
2. **Never run `normalize_text` on validation/test.** It changes string length and therefore invalidates the gold character offsets. It is only safe on `train_unlabeled`, where labels are derived *after* normalization.
3. **`spans_to_character_labels` uses `text.find(span)`** — labels only the *first* occurrence and silently drops non-verbatim spans. **Measured over all 41,601 spans:** 5.98% unmatched against raw dataset text (4.93% against normalized text — the 1.05pp difference is normalization damage), 1.67% occur more than once and only the first is labelled. Fix = all-occurrence matching + a whitespace/quote-normalized fallback. *Note: `find` is self-correcting for leading/trailing whitespace stripping, so offsets in the saved dataset are correct — this is pure recall loss, not misalignment.*
4. **Threshold — RESOLVED, now 0.5.** Use `soft >= 0.5` (majority-or-tie), **not** `> 0.5`. The official gold rule is strictly `>`, but gold has ~3 annotators where `>0.5` means ≥2/3; our ensemble is **even-sized (6 votes)**, so `>0.5` demands 4/6 — stricter than a majority. Measured silver positive-char rates: `>=0.5` → **41.0%**, `>0.5` → 29.9%, `>=0.7` (old) → 21.1%. Gold is **39.6%** (val) / **41.5%** (test). `>=0.5` matches gold to within a point.
5. **Soft labels are currently computed but unused in training.** The model trains on hard labels with cross-entropy, so it has nothing optimizing ρ. To score ρ: take `softmax(logits)[:, 1]` per token and broadcast to that token's character range via offset mapping. Soft-label (distillation / soft cross-entropy) training is the natural next step.
6. **Evaluation gap:** the current eval loop scores token-level predictions on a split of our own silver data. That measures agreement with the LLM annotators, not with humans. Real evaluation = official IoU + ρ against `validation` / `test` gold.
7. **Empty-prediction handling.** Items with no hallucination need the special-case rule (§2) or the IoU averages will be wrong.
8. **mBERT is a weak base per the paper** (BERT-family underperformed; XLM-R was the baseline). `xlm-roberta-base`/`large` is a cheap, likely-positive swap since the tokenizer/offset code is identical.
9. **Tavily's `include_answer="basic"` gives a short snippet, not the Wikipedia page.** The dataset's own reference is `wikipedia_url`. Given the paper's finding that retrieval quality dominates, fetching the actual Wikipedia article (available for val/test) is likely the highest-leverage improvement available.
10. **Secrets:** keys live in `.env`. `research/BERT_classifier.ipynb` cell 15 has a Tavily key in plaintext — strip it before committing.

---

## 6. Status — as of 2026-07-28

**Done:** annotation complete (3,351/3,351); aggregation at threshold 0.5; `train.ipynb` fully wired (tokenize → train → official scorer) and run end to end; official IoU/ρ scorer implemented and verified line-by-line against the paper's §4 definitions.

### 6.1 Results from the completed run

Trained mBERT (`bert-base-multilingual-uncased`) on 3,285 silver rows, evaluated per epoch against **gold validation** (no `train_test_split` — see §6.2).

| epoch | eval_loss | precision | recall | f1 |
|---|---|---|---|---|
| 1 | 0.709 | 0.586 | 0.334 | 0.4256 |
| 2 | 0.756 | 0.553 | 0.456 | **0.5000** ← best |
| 3 | 0.948 | 0.539 | 0.371 | 0.4395 |
| 4 | 1.112 | 0.537 | 0.428 | 0.4760 |
| 5 | 1.203 | 0.534 | 0.448 | 0.4869 |

**Use `num_train_epochs=2`.** Epochs 3–5 strictly hurt. `eval_loss` rises while f1 improves — cross-entropy punishes confidence, f1 scores decisions; this is why `metric_for_best_model="f1"` matters (selecting on loss picks epoch 1).

**Official metrics on test** (earlier 0.7-threshold checkpoint — rerun after the 0.5 retrain):

| | your macro | 43-team mean | best system | human ceiling | mark-all |
|---|---|---|---|---|---|
| **IoU** | 0.318 | 0.43 | 0.65 | 0.72 | 0.345 |
| **ρ** | 0.367 | 0.395 | 0.65 | — | 0.013 |

- IoU below the team mean in **0/14** languages; below `mark-all` overall.
- **ρ above the team mean in 6/14** (ZH +0.093, EN +0.092, SV +0.074, ES +0.050, DE +0.016, AR +0.014) and only 0.028 off the mean overall.
- Smallest IoU gaps: ES, EN, SV, ZH, DE (mostly trained languages). Largest: HI, FI, CS, FA (no training data). **Anomaly worth investigating: FR is 56% of training data yet sits at −0.161.**

### 6.2 Split policy (decided, do not revert)

No `train_test_split`. **Train** = silver `train_unlabeled`; **eval each epoch** = gold `validation` (499); **test** = gold `test` (1,902), held to the end. This means checkpoint selection is driven by *human* labels, not by agreement with our own LLM annotators.

### 6.3 Root-cause diagnosis — the model has no evidence to reason from

Ruled out by measurement:
- **Label→token alignment is fine.** Round-trip IoU (gold chars → token labels → chars) = **0.909**, and near-identical for mBERT-uncased / mBERT-cased / XLM-R. Not a mask or offset bug.
- **Silver labels are sound.** LLM-ensemble agreement by the paper's eq. (1) = **0.599**, comparable to *human* IAA on EN (0.49), ES (0.51), ZH (0.58).
- **Labels are definitely used.** Loss with real labels 0.2323 vs shuffled 0.6522. Model fits train silver at f1 **0.907** but gold val at **0.505** — it learned the labels; they don't transfer.
- **Generator logits are useless here.** AUC of (low logit → hallucinated) = **0.526** (0.5 = random). Confirmed independently by team *keepitsimple*, who found uncertainty alone insufficient without retrieval. Keep as a negative ablation.

**The actual cause:** `tokenize_and_align_labels` feeds **only `model_output_text`**. No question, no reference. The annotator ensemble saw both; the model sees neither.

Proof it is unlearnable as posed: **74.3% of token instances have a token string that appears with BOTH labels** in gold. The ceiling for *any* text-only model (perfect memorisation of the best label per token) is **76.13%**; mBERT is at 63.83%; majority baseline is 59.03%.

This is exactly the official XLM-R baseline — `participant_kit/baseline_model.py` also does `tokenizer(examples['model_output_text'], ...)`, answer-only — which the paper says "ranks extremely low."

### 6.4 What the literature says (searched 2026-07-28)

Paper §6 significance tests over 2,618 submissions:
- **Prompt-based vs fine-tuned: NO significant IoU difference.** Fine-tuning only helps **ρ** (p < 0.002).
- **RAG: significantly higher IoU (p < 10⁻⁵⁹, f = 69.80%) and ρ (p < 10⁻³⁹).** The dominant effect in the whole analysis.
- Teams relying mainly on provided data scored lower (p < 10⁻²¹) — doesn't apply to us; we built our own silver labels.

**Our profile (ρ at field average, IoU far below) is precisely the fine-tuning-without-RAG signature.** That's the framing for the report.

Key teams:
- **ATLANTIS** ([arXiv:2508.05179](https://arxiv.org/pdf/2508.05179)) — closest analogue. Fine-tuned XLM-R-large on synthetic data. Input = **question + top-1 Wikipedia chunk + answer**. Ablation: **without retrieval 0.39 → with retrieval 0.49** (EN/DE/ES/FR), "better in all languages, without exception." Also: question at the **beginning** works best; and they found the model "globally under-confident" and **lowered the decision threshold** to raise IoU — matching our own sweep, which peaked at 0.10, the lowest value tested. *Our 0.327 on those same four languages vs their no-retrieval 0.39, with mBERT-base and 3.3k samples against XLM-R-large and 48k.*
- **UCSC** ([arXiv:2505.03030](https://arxiv.org/abs/2505.03030)) — **#1 overall, no fine-tuning at all.** Retrieve → GPT-4o identifies unsupported content → map back via substring/LLM/edit-distance → MiPROv2 prompt optimization. IoU 0.61.
- **MSA** ([arXiv:2505.20880](https://arxiv.org/pdf/2505.20880)) — same weak-labeling idea as ours, but **did not fine-tune on the weak labels**; ran the ensemble directly at inference over (question, reference, answer).
- **PsiloQA** ([arXiv:2510.04849](https://arxiv.org/abs/2510.04849), [github.com/s-nlp/psiloqa](https://github.com/s-nlp/psiloqa), [HF s-nlp/PsiloQA](https://huggingface.co/datasets/s-nlp/PsiloQA)) — our pipeline at 20× scale. **63,792 train / 3,355 val / 2,897 test, 14 languages**, free download. Schema maps directly: `question`→`model_input`, `wiki_passage`→the context we lack, `llm_answer`→`model_output_text`, `labels` = **identical `[[start,end]]` format** to our `hard_labels`. Finding: fine-tuned encoders beat LLM prompting and uncertainty quantification.

### 6.5 Code located for reuse

- **`participant_kit/`** — downloaded, extracted to scratchpad; also at `https://a3s.fi/mickusti-2007780-pub/participant_kit.zip` (scorer alone: `.../scorer.py`). Contains the official XLM-R baseline. *Note it labels a token only if fully inside a gold span (`start >= label_start and end <= label_end`); ours labels on any overlap, slightly more recall-friendly.*
- **PsiloQA `train_script.py`** — the fine-tuning script to adapt. Uses `lettucedetect` as a library (`pip install lettucedetect`); still fine-tunes from a base checkpoint, not an imported pretrained detector. The swap is ~10 lines in `build_samples()`:
  ```python
  labels = [{"start": int(s), "end": int(e)} for s, e in row["hard_labels"]]
  HallucinationSample(prompt=row["model_input"],        # ← reference + question go here
                      answer=row["model_output_text"],
                      labels=labels, split=split_name, task_type="qa",
                      dataset="verillm", language=row["lang"])
  ```
  Advantages: `prompt=` is a real input field; `max_length=8192` (ModernBERT/EuroBERT) so a full Wikipedia passage fits, unlike mBERT's 512; base model is a `--model-name` flag. For multilingual use **EuroBERT-210m/610m or mmBERT-base**, not ModernBERT-base (English-only).
- **[LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect)** (MIT) — the underlying trainer; input is context+question+answer, output is character-offset spans.

---

## 7. RESUME HERE — next session

Nothing below has been implemented. **Ask before editing project files.**

### Step 1 — pair encoding (free, ~20 lines, `train.ipynb`)
Put the question into the input. This is the whole diagnosis in one change.
```python
encoding = tokenizer(example["model_input"], example["model_output_text"],
                     truncation="only_first", max_length=512,
                     return_offsets_mapping=True)
seq = encoding.sequence_ids()
# label only where seq[i] == 1; -100 on question tokens and specials
```
`truncation="only_first"` is essential — it cuts the question, never the answer, so character offsets stay valid. ATLANTIS ablated placement: **question at the beginning wins**. Also set `num_train_epochs=2`.

### Step 2 — add the reference (free, the change that should actually move IoU)
- **Train:** `reference_answer` already cached in `annotations.jsonl`, 3,351/3,351.
- **Val/test:** `wikipedia_url` populated on **all** 499 + 1,902 rows → fetch via the **free, unmetered MediaWiki API**. This costs *no Tavily quota*. (§5.9's "highest-leverage change" is cheaper than it looked.)
- Caveat to validate on a small subset first: train refs are Tavily snippets, val/test would be Wikipedia articles — a length/style mismatch that may blunt the gain.

### Step 3 — lower the decision threshold
Our sweep peaked at **0.10**, the lowest value tested and it never turned over — retest below 0.1. ATLANTIS independently reported the same under-confidence. Tune on validation only, never test; report the official 0.5 as the headline and the sweep as analysis.

### Step 4 — scale up (optional)
Adopt PsiloQA's `train_script.py` (§6.5) and/or concatenate its 63,792 rows with ours. It ships `wiki_passage`, so it fixes context and data scarcity together, for zero API calls. Covers CA/CS/EU/FA, where we score worst.

### Still open from before
- Soft labels are computed but **unused in the loss** — nothing optimises ρ. Soft cross-entropy against the vote fraction is the fix.
- Re-run the official scorer on the 0.5-threshold checkpoint; the §6.1 test numbers are from the older 0.7 model.
- Fix `text.find` label loss (§5.3): all-occurrence + normalized fallback.
- Swap base model → XLM-R / EuroBERT / mmBERT.
- Port notebooks into `src/VeriLLM/` (still empty scaffolding).
- Strip the hardcoded Tavily key from `BERT_classifier.ipynb` before any push.

### Report material already computed (all free, all in hand)
- Per-language IoU/ρ vs the 43-team mean, best system, human ceiling, and mark-all (§6.1).
- LLM-ensemble agreement 0.599 vs human IAA 0.49–0.87 via the paper's eq. (1).
- Threshold ablation 0.5 / 0.6 / 0.7 with positive-char rates against gold.
- Negative ablation: generator logits, AUC 0.526.
- Text-only ceiling analysis: 74.3% ambiguous tokens, 76.13% ceiling vs 63.83% achieved.
- Round-trip alignment 0.909 across three tokenizers.
- **Framing:** *"a distilled no-retrieval token classifier reaches 0.318 IoU / 0.367 ρ — below the 0.43 field mean on IoU but within 0.03 on ρ, with the deficit concentrated in languages absent from training. This reproduces the effect the task paper attributes to fine-tuning while isolating the absence of the effect it attributes to RAG."*
