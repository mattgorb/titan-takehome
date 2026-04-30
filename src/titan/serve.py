"""FastAPI service for the fine-tuned model.

Run:
  uvicorn titan.serve:app --host 0.0.0.0 --port 8000

Loads the latest adapter (runs/LATEST → runs/<id>/adapter) at startup.
Pin a specific run with `TITAN_ADAPTER=runs/<id>/adapter`.
Override the config path with `TITAN_CONFIG=...`.
"""
from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from titan.config import Config, load_config
from titan.inference import LoadedModel, generate, load


CONFIG_PATH = os.environ.get("TITAN_CONFIG", "configs/default.yaml")


class AskRequest(BaseModel):
    question: str
    context: Optional[str] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None


class AskResponse(BaseModel):
    answer: str
    run_id: str


app = FastAPI(title="Titan banking SLM", version="0.1.0")
_state: dict = {}


def _resolve_adapter(cfg: Config) -> tuple[str, Path]:
    pin = os.environ.get("TITAN_ADAPTER")
    if pin:
        path = Path(pin)
        return path.parent.name, path
    latest = Path(cfg.runs_dir) / "LATEST"
    if not latest.exists():
        raise RuntimeError("No trained adapter found. Train a model first or set TITAN_ADAPTER.")
    run_id = latest.read_text().strip()
    return run_id, Path(cfg.runs_dir) / run_id / "adapter"


@app.on_event("startup")
def _startup() -> None:
    cfg = load_config(CONFIG_PATH)
    run_id, adapter_path = _resolve_adapter(cfg)
    lm = load(cfg, adapter_path=adapter_path)
    _state["cfg"] = cfg
    _state["lm"] = lm
    _state["run_id"] = run_id
    _state["loaded_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"


@app.get("/health")
def health() -> dict:
    cfg: Optional[Config] = _state.get("cfg")
    if cfg is None:
        raise HTTPException(503, "model not loaded")
    return {
        "status": "ok",
        "base_model": cfg.model.base,
        "adaptation": "lora-peft",
        "run_id": _state["run_id"],
        "loaded_at": _state["loaded_at"],
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    lm: Optional[LoadedModel] = _state.get("lm")
    if lm is None:
        raise HTTPException(503, "model not loaded")
    out = generate(
        lm,
        req.question,
        input_text=req.context,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )
    return AskResponse(answer=out, run_id=_state["run_id"])
