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

TODO — paste from `runs/<id>/eval/metrics.json` after a real run, plus 1-2 paragraphs of honest assessment (where it works, where it fails).

## Limitations

- CPU only — long training runs require careful subset / epoch budgets.
- Refusal classifier is keyword-heuristic.
- Single-turn only; no conversation history.
- Surface-level metrics (ROUGE / BERTScore / semsim) reward overlap with the reference, not factual correctness.

## What I'd do with more time

TODO.
