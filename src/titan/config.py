"""Typed configuration loaded from YAML (see configs/default.yaml)."""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class ModelCfg(BaseModel):
    base: str = "Qwen/Qwen2.5-0.5B"
    device: str = "cpu"
    dtype: str = "float32"
    max_length: int = 512


class DataCfg(BaseModel):
    hf_dataset: str = "gbharti/finance-alpaca"
    train_size: int = 5000
    val_size: int = 500
    test_size: int = 500
    curate: bool = True


class LoraCfg(BaseModel):
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])


class TrainCfg(BaseModel):
    epochs: int = 1
    batch_size: int = 2
    grad_accum: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    max_steps: int = -1


class EvalCfg(BaseModel):
    num_samples: int = 200
    num_qualitative: int = 15
    metrics: list[str] = Field(default_factory=lambda: ["rouge", "bertscore", "semsim"])
    semsim_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    refusal_prompts: str = "configs/refusal_prompts.yaml"
    # Generation overrides for eval. Decoupled from serve so production serving
    # can stay sample-based while eval runs greedy + tighter token cap for speed.
    max_new_tokens: int = 128
    temperature: float = 0.0   # 0 = greedy


class ServeCfg(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9


class Config(BaseModel):
    seed: int = 42
    run_id: Optional[str] = None
    runs_dir: str = "runs"
    data_dir: str = "data"
    model: ModelCfg = Field(default_factory=ModelCfg)
    data: DataCfg = Field(default_factory=DataCfg)
    lora: LoraCfg = Field(default_factory=LoraCfg)
    train: TrainCfg = Field(default_factory=TrainCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)
    serve: ServeCfg = Field(default_factory=ServeCfg)


def load_config(path: str | Path) -> Config:
    """Load and validate config from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return Config(**raw)
