"""Run quantitative + qualitative evaluation on the held-out test split.

Produces:
  runs/<run_id>/eval/metrics.json  — quantitative scores + refusal stats
  runs/<run_id>/eval/report.md     — qualitative samples + refusal samples

CLI:
  python -m titan.eval.run --config configs/default.yaml [--run-id <id>]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from titan.config import load_config
from titan.seed import set_seed
from titan.tracking import append_metrics, get_latest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run-id", default=None, help="Run dir to evaluate; defaults to runs/LATEST")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    run_id = args.run_id or get_latest(cfg.runs_dir)
    rd = Path(cfg.runs_dir) / run_id
    eval_dir = rd / "eval"
    eval_dir.mkdir(exist_ok=True)
    print(f"[eval] run={run_id} dir={rd}")

    # Load model + adapter
    from titan.inference import generate, load
    lm = load(cfg, adapter_path=rd / "adapter")

    # Load test split
    from datasets import load_from_disk
    splits = load_from_disk(str(Path(cfg.data_dir) / "processed"))
    test = splits["test"].select(range(min(cfg.eval.num_samples, len(splits["test"]))))

    import time
    preds: list[str] = []
    refs: list[str] = []
    samples: list[dict] = []
    print(f"[eval] generating on {len(test)} test examples...")
    t_total_start = time.time()
    for i, ex in enumerate(test):
        t0 = time.time()
        out = generate(lm, ex["instruction"], ex.get("input"))
        dt = time.time() - t0
        preds.append(out)
        refs.append(ex["output"])
        if i < cfg.eval.num_qualitative:
            samples.append({
                "instruction": ex["instruction"],
                "input": ex.get("input", "") or "",
                "reference": ex["output"],
                "prediction": out,
            })
        # First example often takes much longer on MPS (shader compile);
        # log it explicitly so the user can tell hang from slow-warmup.
        if i == 0 or (i + 1) % 5 == 0:
            print(f"  [{i + 1}/{len(test)}] {dt:.1f}s  (avg {(time.time() - t_total_start) / (i + 1):.1f}s)")

    # Quantitative metrics
    from titan.eval.metrics import bertscore, rouge_l, semsim
    metrics: dict = {}
    if "rouge" in cfg.eval.metrics:
        metrics.update(rouge_l(preds, refs))
    if "bertscore" in cfg.eval.metrics:
        metrics.update(bertscore(preds, refs))
    if "semsim" in cfg.eval.metrics:
        metrics.update(semsim(preds, refs, cfg.eval.semsim_model))

    # Refusal eval
    from titan.eval.refusal import load_refusal_prompts, score as refusal_score
    refusal_prompts = load_refusal_prompts(cfg.eval.refusal_prompts)
    refusal_outputs: list[tuple[dict, str]] = []
    for rp in refusal_prompts:
        out = generate(lm, rp["question"])
        refusal_outputs.append((rp, out))
    metrics["refusal"] = refusal_score(refusal_outputs)

    # Persist
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _write_report(eval_dir / "report.md", metrics, samples, refusal_outputs)
    append_metrics(rd, {"phase": "eval", **{k: v for k, v in metrics.items() if k != "refusal"}})
    print(json.dumps(metrics, indent=2))


def _write_report(
    path: Path,
    metrics: dict,
    samples: list[dict],
    refusal_outputs: list[tuple[dict, str]],
) -> None:
    quant = {k: v for k, v in metrics.items() if k != "refusal"}
    lines: list[str] = [
        "# Evaluation Report",
        "",
        "## Quantitative metrics",
        "",
        "```json",
        json.dumps(quant, indent=2),
        "```",
        "",
        "## Refusal evaluation (Tier 2)",
        "",
        "```json",
        json.dumps(metrics.get("refusal", {}), indent=2),
        "```",
        "",
        "## Qualitative samples",
        "",
    ]
    for i, s in enumerate(samples, 1):
        lines += [f"### Sample {i}", f"**Instruction:** {s['instruction']}", ""]
        if s.get("input"):
            lines += [f"**Context:** {s['input']}", ""]
        lines += [
            f"**Reference:** {s['reference']}",
            "",
            f"**Prediction:** {s['prediction']}",
            "",
            "---",
            "",
        ]

    lines += ["## Refusal samples", ""]
    for meta, out in refusal_outputs:
        lines += [
            f"### {meta['id']} ({meta['category']})",
            f"**Q:** {meta['question']}",
            "",
            f"**A:** {out}",
            "",
            "---",
            "",
        ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
