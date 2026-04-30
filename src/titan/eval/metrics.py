"""Wrappers over rouge, BERTScore, and sentence-transformer cosine similarity.

These are surface-level metrics — they reward lexical or embedding-space
overlap with the reference, NOT factual correctness. The README should be
explicit about that limitation.
"""
from __future__ import annotations
from functools import lru_cache


def rouge_l(preds: list[str], refs: list[str]) -> dict:
    from rouge_score import rouge_scorer

    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [sc.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
    return {"rougeL_f1": sum(scores) / max(1, len(scores))}


def bertscore(preds: list[str], refs: list[str], model_type: str = "distilbert-base-uncased") -> dict:
    from bert_score import score as bs

    _, _, F = bs(preds, refs, model_type=model_type, lang="en", verbose=False)
    return {"bertscore_f1": float(F.mean())}


@lru_cache(maxsize=2)
def _semsim_model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def semsim(preds: list[str], refs: list[str], model_name: str) -> dict:
    import numpy as np

    m = _semsim_model(model_name)
    pe = m.encode(preds, normalize_embeddings=True)
    re = m.encode(refs, normalize_embeddings=True)
    sims = (pe * re).sum(axis=1)
    return {"semsim": float(np.mean(sims))}
