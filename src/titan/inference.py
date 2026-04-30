"""Load Qwen2.5-0.5B + LoRA adapter once and generate. Shared by eval and serve."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from titan.config import Config
from titan.data.format import render_for_inference


@dataclass
class LoadedModel:
    tokenizer: object
    model: object
    config: Config


_DTYPES = {"float32": "float32", "bfloat16": "bfloat16", "float16": "float16"}


def load(cfg: Config, adapter_path: Optional[str | Path] = None) -> LoadedModel:
    """Load base model + tokenizer; optionally apply a LoRA adapter on top."""
    import time
    t = time.time()
    print(f"[inference] importing torch / transformers / peft...", flush=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[inference] imports done ({time.time() - t:.1f}s); loading {cfg.model.base}...", flush=True)

    dtype = getattr(torch, _DTYPES.get(cfg.model.dtype, "float32"))
    tok = AutoTokenizer.from_pretrained(cfg.model.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # `dtype=` replaces deprecated `torch_dtype=` in transformers >= 4.45.
    model = AutoModelForCausalLM.from_pretrained(cfg.model.base, dtype=dtype)
    print(f"[inference] base model loaded ({time.time() - t:.1f}s)", flush=True)

    # Apply adapter BEFORE moving to device — otherwise the adapter loads on
    # CPU and forward passes silently fall back to CPU on every step.
    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path))
        print(f"[inference] adapter applied ({time.time() - t:.1f}s)", flush=True)

    model = model.to(cfg.model.device)
    model.eval()
    print(f"[inference] ready on {cfg.model.device}, dtype={cfg.model.dtype} ({time.time() - t:.1f}s total)", flush=True)
    return LoadedModel(tokenizer=tok, model=model, config=cfg)


def generate(
    lm: LoadedModel,
    instruction: str,
    input_text: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> str:
    """Run a single inference and return only the assistant's text (prompt stripped)."""
    import torch

    prompt = render_for_inference(lm.tokenizer, instruction, input_text)
    enc = lm.tokenizer(prompt, return_tensors="pt").to(lm.config.model.device)

    temp = temperature if temperature is not None else lm.config.serve.temperature
    with torch.inference_mode():
        out = lm.model.generate(
            **enc,
            max_new_tokens=max_new_tokens or lm.config.serve.max_new_tokens,
            do_sample=temp > 0,
            temperature=temp if temp > 0 else 1.0,
            top_p=top_p if top_p is not None else lm.config.serve.top_p,
            pad_token_id=lm.tokenizer.eos_token_id,
        )

    prompt_len = enc.input_ids.shape[1]
    new_ids = out[0, prompt_len:]
    return lm.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
