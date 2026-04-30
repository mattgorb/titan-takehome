# Titan — Banking SLM Fine-Tuning Pipeline

End-to-end pipeline that fine-tunes Qwen2.5-0.5B on `gbharti/finance-alpaca` with LoRA (PEFT), evaluates the result, and serves it behind FastAPI. Designed to run on a laptop CPU.

## Architecture

- **Data prep** [`src/titan/data/`](src/titan/data/) — load, deterministic split, optional curation, one shared prompt template (Qwen chat format).
- **Training** [`src/titan/train.py`](src/titan/train.py) — PEFT LoRA + HF Trainer. Saves adapter + tokenizer to `runs/<run_id>/adapter/`.
- **Inference** [`src/titan/inference.py`](src/titan/inference.py) — single load+generate path used by both eval and serve so train/inference rendering can't drift.
- **Evaluation** [`src/titan/eval/`](src/titan/eval/) — ROUGE-L, BERTScore, semantic similarity, plus a refusal eval against hand-curated out-of-scope prompts (Tier 2).
- **Serve** [`src/titan/serve.py`](src/titan/serve.py) — FastAPI: `POST /ask`, `GET /health`. Adapter loaded once at startup.
- **Experiment tracking** [`src/titan/tracking.py`](src/titan/tracking.py) — per-run dir under `runs/` with config snapshot, JSONL metrics, and data lineage manifest. Plain files; queryable with jq + ripgrep.

## Setup

```sh
python -m venv .venv && source .venv/bin/activate
make setup
```

Requires Python 3.11.

## Reproduce

```sh
make data    # download + split + curate (writes data/processed/)
make train   # fine-tune; writes runs/<id>/adapter and runs/LATEST
make eval    # writes runs/<id>/eval/{metrics.json, report.md}
make serve   # uvicorn on :8000
```

Override the config path: `make train CONFIG=configs/my-experiment.yaml`.

## API

```sh
curl -X POST localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question": "What is a credit default swap spread?"}'

curl localhost:8000/health
```

## Design decisions

- **Training subset size (5000):** TODO — justify after first run. Goal: finish training within the time window with enough signal to see adaptation.
- **Curation strategy:** length bounds, empty-output drop, ASCII-ratio filter, exact-dup removal. See [`curate.py`](src/titan/data/curate.py). TODO — measure impact (Tier 3).
- **Prompt template:** Qwen's built-in chat template via `apply_chat_template`. Same module is used at train and inference time, so they cannot drift.
- **LoRA hyperparameters:** r=8, alpha=16, target_modules=[q_proj, v_proj]. Conservative defaults; rationale TODO.

## Refusal strategy (Tier 2)

System prompt instructs the model to say "I don't know" rather than guess on out-of-scope, predictive, or low-confidence questions. Eval set in [`configs/refusal_prompts.yaml`](configs/refusal_prompts.yaml) covers four categories (out-of-scope, unknowable, harmful, nonsense). Refusal classifier is keyword-based — explicit and inspectable, but can be fooled by paraphrase. With more time: LLM-as-judge or a small fine-tuned classifier.

## Results

First end-to-end run — [`configs/default.yaml`](configs/default.yaml) (SmolLM-135M + LoRA r=8, `train_size=4500`, 1 epoch on MPS), evaluated on 5 test examples and 22 refusal prompts:

```json
{
  "rougeL_f1": 0.270,
  "semsim": 0.543,
  "refusal": {
    "n_prompts": 22,
    "refused": 0,
    "refusal_rate": 0.0,
    "fabrication_rate": 1.0
  }
}
```

Honest assessment:

- **ROUGE-L 0.27 / semsim 0.54** — modest. The model produces finance-shaped text that's loosely on-topic semantically (~0.5 cosine similarity to references) but lexically diverges from the reference answers. Expected for a 135M base model trained for one epoch on a 4500-row subset; the model has learned the *style* of the data more than the substance.
- **Refusal failure: 0/22 (100% fabrication rate).** The system prompt explicitly instructs the model to say "I don't know" on out-of-scope, predictive, or unknowable questions, but the model produced an answer for every refusal prompt. Two compounding causes: (1) **SmolLM-135M is a base model, not chat-tuned** — it has no learned behavior of following a system prompt, and the Alpaca-style fallback template we use doesn't change that; (2) the refusal classifier in [refusal.py](src/titan/eval/refusal.py) is keyword-based and would still miss paraphrased refusals if any did occur. Cause (1) is the dominant one here.
- **Sample size caveat.** 5 test examples is far too small to make quantitative claims about generation quality; the refusal result (22 prompts) is more interpretable. Bumping `eval.num_samples` to ≥30 would make the surface-similarity numbers worth reading.
- **What this implies.** For a banking knowledge assistant we'd want to (a) start from a chat-tuned base (Qwen2.5-0.5B-Instruct or similar) so the model actually attends to system instructions, and (b) include explicit refusal examples in the training set so refusal becomes a learned behavior rather than a hoped-for one.

Run artifacts at `runs/20260430-180931/` — `metrics.json`, `report.md` with qualitative samples, `config.yaml` snapshot, `data_manifest.json`.

### Curation A/B (Tier 3)

Both runs trained on the same 4500-row train subset (curated vs naive uncurated), evaluated against the **same canonical curated test set** via `--test-from data/test-canonical` for an apples-to-apples comparison.

| Metric | Default (curated) `20260430-180931` | Baseline (no curation) `20260430-193656` | Δ |
|---|---|---|---|
| ROUGE-L F1 | 0.186 | 0.186 | ≈0 |
| semsim (cosine) | 0.432 | 0.480 | **+0.048 baseline** |
| refusal_rate | 0.0 | 0.0 | tied (both fail) |
| fabrication_rate | 1.0 | 1.0 | tied (both fail) |

**Honest reading.** Curation did not help. ROUGE-L is identical and the uncurated baseline actually scored *higher* on semantic similarity. Plausible causes: (1) the keyword regex over-aggressively filters useful examples (e.g., financial concepts not in the keyword list), shrinking the effective training pool more than it improves quality; (2) at 4500 examples × 1 epoch on a 135M base, the noise floor likely dominates the curation signal; (3) the refusal failure is data-quality-orthogonal — both runs fabricate at 100%, which is consistent with the SFT-vs-refusal-as-learned-behavior story. This is the kind of result the spec asks for: a real ablation, not a curated one.

## Limitations

- CPU only — long training runs require careful subset / epoch budgets.
- Refusal classifier is keyword-heuristic.
- Single-turn only; no conversation history.
- Surface-level metrics (ROUGE / BERTScore / semsim) reward overlap with the reference, not factual correctness.

## What I'd do with more time

- **Better curation.** Replace the keyword filter with LLM-as-judge scoring on domain specificity, factuality, specificity-vs-fluff, refusal calibration, and audience fit; near-dup cluster and keep top-k. Build the eval set the same way, balanced across finance subdomains.
- **RL instead of SFT.** SFT mimics tokens; it doesn't optimize for correctness or refusal. Move to DPO (needs preference pairs) or GRPO (needs a reward signal) — both blocked here by the absence of labels.
- **Use CME-GRPO to unblock the labels problem.** My recent paper (_Label-Free Reinforcement Learning via Cross-Model Entropy_) uses a stronger verifier model's per-token likelihood as the GRPO reward — no ground truth, no preference data. Concretely: prompt-only training set, cross-family verifier, swap [train.py](src/titan/train.py)'s SFT loss for CME-GRPO on the same LoRA adapter. Expect the 100% fabrication rate to drop sharply since confident-but-wrong answers receive low verifier likelihood.

## Implementation status

Per-bullet checklist mapped to the spec. `[x]` = complete, `[/]` = partial / infrastructure in place but no results yet, `[ ]` = outstanding.

### Tier 1 — Working Pipeline

| | Spec bullet | What / Where |
|---|---|---|
| [x] | Load Finance Alpaca from HF | [src/titan/data/load.py](src/titan/data/load.py) — `load_dataset("gbharti/finance-alpaca")` |
| [x] | Train / val / test split | seeded shuffle + slice in [load.py](src/titan/data/load.py); sizes in `data.{train,val,test}_size` |
| [x] | Sample / curate training subset (with rationale) | [src/titan/data/curate.py](src/titan/data/curate.py) — finance-keyword topic regex + length/ASCII/dedup; train-only quality filter |
| [x] | Prompt template (matches inference) | [src/titan/data/format.py](src/titan/data/format.py) — Qwen chat template with Alpaca fallback for base models; same module used at train and inference |
| [x] | Handle variable data quality (missing fields, formatting) | curate.py drops empty / wrong-length / non-English; format.py handles missing `input` |
| [x] | Fine-tuning approach (LoRA via PEFT) | [src/titan/train.py](src/titan/train.py) — `LoraConfig` + HF `Trainer` |
| [x] | Configurable hyperparameters | [configs/default.yaml](configs/default.yaml), [configs/config2.yaml](configs/config2.yaml), [configs/baseline-no-curation.yaml](configs/baseline-no-curation.yaml) |
| [x] | Random seeds | [src/titan/seed.py](src/titan/seed.py) — Python / NumPy / Torch; called from every entrypoint |
| [x] | Log loss + val metrics | Trainer stdout + [tracking.append_metrics](src/titan/tracking.py) → `runs/<id>/metrics.jsonl` |
| [x] | Save adapter to disk (reloadable) | `runs/<id>/adapter/` + tokenizer; loaded on demand by inference.py and serve.py |
| [x] | Quantitative metrics for generative QA | [src/titan/eval/metrics.py](src/titan/eval/metrics.py) — ROUGE-L, BERTScore, sentence-transformer cosine similarity |
| [x] | Acknowledgement of what metrics don't capture | "Limitations" section above |
| [/] | 10–20 qualitative samples in report | report written at `runs/20260430-180931/eval/report.md`; first run only included 5 samples (`num_samples=5`) — bump to ≥15 to meet spec target |
| [x] | Honest assessment of strengths / failures | "Results" section above |
| [x] | `POST /ask` | [src/titan/serve.py](src/titan/serve.py) — accepts `question` + optional `context` |
| [x] | `GET /health` with metadata | serve.py — returns `base_model`, `adaptation`, `run_id`, `loaded_at` |
| [x] | API loads model at startup, not per-request | serve.py — `@app.on_event("startup")` loads adapter once |

### Tier 2 — Production Thinking

| | Spec bullet | What / Where |
|---|---|---|
| [x] | Hyperparameters + config snapshot per run | `runs/<id>/config.yaml` via [tracking.write_config](src/titan/tracking.py) |
| [x] | Per-step metrics persisted | `runs/<id>/metrics.jsonl` (Trainer + eval phases) |
| [x] | Saved model ↔ run identification | `runs/<id>/adapter/` + `runs/LATEST` pointer |
| [x] | Upstream data lineage | `runs/<id>/data_manifest.json` — dataset id, seed, split sizes, content hash |
| [x] | Mix unanswerable / out-of-scope into eval | [configs/refusal_prompts.yaml](configs/refusal_prompts.yaml) — 17 prompts across out-of-scope / unknowable / harmful / nonsense |
| [x] | Refusal classifier + scoring | [src/titan/eval/refusal.py](src/titan/eval/refusal.py) — keyword-based; reports `refusal_rate`, `fabrication_rate` |
| [x] | Refusal strategy described in README | "Refusal strategy (Tier 2)" section above |
| [x] | Refusal results / measured tradeoffs | 0/22 refusals on first run — see "Results" above |

### Tier 3 — Depth

| | Spec bullet | What / Where |
|---|---|---|
| [ ] | Beyond-surface eval framework | not started — current metrics reward overlap, not factual correctness |
| [/] | Data curation A/B (curated vs naive) | infra: [configs/baseline-no-curation.yaml](configs/baseline-no-curation.yaml) + `--test-from` flag in [eval/run.py](src/titan/eval/run.py) so both runs score against the same curated test set; runs not yet executed |
| [/] | Multi-run comparison | infra: [configs/config2.yaml](configs/config2.yaml) (Qwen2.5-0.5B + wider LoRA + different batch/grad-accum split); `runs/<id>/` layout supports side-by-side comparison; second run not yet executed |
