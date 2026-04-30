"""Fine-tune Qwen2.5-0.5B on the finance-alpaca subset using LoRA via PEFT.

CLI:
  python -m titan.train --config configs/default.yaml
"""
from __future__ import annotations
import argparse
from pathlib import Path

from titan.config import load_config
from titan.seed import set_seed
from titan.tracking import (
    append_metrics,
    new_run_id,
    run_dir,
    set_latest,
    write_config,
    write_data_manifest,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    run_id = cfg.run_id or new_run_id()
    rd = run_dir(cfg.runs_dir, run_id)
    write_config(rd, cfg.model_dump(mode="json"))
    print(f"[run] id={run_id} dir={rd}")

    # 1. Load (or rebuild) the splits
    from datasets import load_from_disk

    from titan.data.format import render_for_training
    from titan.data.load import load_and_split, manifest

    splits_dir = Path(cfg.data_dir) / "processed"
    if splits_dir.exists():
        splits = load_from_disk(str(splits_dir))
        print(f"[data] loaded splits from {splits_dir}")
    else:
        splits = load_and_split(cfg)
        print("[data] built splits in-memory (run `make data` to persist)")
    write_data_manifest(rd, manifest(splits, cfg))

    # 2. Tokenizer + base model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model.base)

    # 3. LoRA wrap
    from peft import LoraConfig, TaskType, get_peft_model

    lora = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # 4. Tokenize. Don't set labels here — DataCollatorForLanguageModeling
    # creates them at batch time and pads with -100 so loss ignores padding.
    def tok_fn(ex):
        rendered = render_for_training(tok, ex)
        return tok(rendered["text"], truncation=True, max_length=cfg.model.max_length)

    train_ds = splits["train"].map(tok_fn, remove_columns=splits["train"].column_names)
    val_ds = splits["validation"].map(tok_fn, remove_columns=splits["validation"].column_names)

    # 5. Trainer
    from transformers import (
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    args_t = TrainingArguments(
        output_dir=str(rd / "checkpoints"),
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.train.batch_size,
        per_device_eval_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        learning_rate=cfg.train.learning_rate,
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay,
        logging_steps=cfg.train.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.train.eval_steps,
        save_steps=cfg.train.save_steps,
        max_steps=cfg.train.max_steps,
        seed=cfg.seed,
        report_to=[],
        use_cpu=(cfg.model.device == "cpu"),
    )
    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    trainer = Trainer(
        model=model,
        args=args_t,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer.train()

    # 6. Save adapter + tokenizer
    adapter_dir = rd / "adapter"
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)

    # 7. Final eval + LATEST pointer
    final = trainer.evaluate()
    append_metrics(rd, {"phase": "train_final", **final})
    set_latest(cfg.runs_dir, run_id)
    print(f"[done] adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
