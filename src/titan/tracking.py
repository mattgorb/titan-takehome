"""Per-run artifact tracking: config, metrics, data manifest, adapter path.

Each run writes to runs/<run_id>/:
  - config.yaml         (snapshot of the config used)
  - metrics.jsonl       (training + eval metrics, append-only)
  - data_manifest.json  (dataset id, split sizes, content hash)
  - adapter/            (saved LoRA adapter, written by train.py)

The directory layout IS the experiment tracker — queryable with jq / ripgrep.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import json

import yaml


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run_dir(runs_dir: str | Path, run_id: str) -> Path:
    p = Path(runs_dir) / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_config(d: Path, config_dict: dict) -> None:
    (d / "config.yaml").write_text(yaml.safe_dump(config_dict, sort_keys=False))


def write_data_manifest(d: Path, manifest: dict) -> None:
    (d / "data_manifest.json").write_text(json.dumps(manifest, indent=2))


def append_metrics(d: Path, metrics: dict) -> None:
    """Append a metrics record to runs/<id>/metrics.jsonl."""
    entry = {"ts": datetime.utcnow().isoformat(timespec="seconds") + "Z", **metrics}
    with (d / "metrics.jsonl").open("a") as out:
        out.write(json.dumps(entry) + "\n")


def set_latest(runs_dir: str | Path, run_id: str) -> None:
    """Pin a convenience pointer so eval/serve can find the most recent run."""
    Path(runs_dir, "LATEST").write_text(run_id)


def get_latest(runs_dir: str | Path) -> str:
    p = Path(runs_dir) / "LATEST"
    if not p.exists():
        raise FileNotFoundError(f"No LATEST pointer at {p}; run training first or pass --run-id")
    return p.read_text().strip()
