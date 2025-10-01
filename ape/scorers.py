from __future__ import annotations

import math
from typing import Iterable, Sequence

from .types import Example


def exact_match(prediction: str, reference: str) -> float:
    """Returns 1.0 if the prediction matches the reference exactly."""

    return float(prediction.strip() == reference.strip())


def rouge_l(prediction: str, reference: str) -> float:
    """A light-weight ROUGE-L implementation for non-learning environments."""

    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    # Longest common subsequence dynamic programming algorithm.
    dp = [[0] * (len(ref_tokens) + 1) for _ in range(len(pred_tokens) + 1)]
    for i in range(1, len(pred_tokens) + 1):
        for j in range(1, len(ref_tokens) + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[-1][-1]
    prec = lcs / len(pred_tokens)
    rec = lcs / len(ref_tokens)
    if prec == 0 or rec == 0:
        return 0.0
    beta = prec / (rec + 1e-12)
    denom = rec + beta * prec
    if denom == 0:
        return 0.0
    return ((1 + beta ** 2) * prec * rec) / denom


def accuracy(predictions: Sequence[str], references: Sequence[str]) -> float:
    if not predictions:
        return 0.0
    total = 0.0
    for pred, ref in zip(predictions, references):
        total += exact_match(pred, ref)
    return total / len(predictions)


def aggregate_scores(scores: Iterable[float]) -> float:
    scores = list(scores)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def evaluate_predictions(
    predictions: Sequence[str],
    examples: Sequence[Example],
    *,
    scorer=exact_match,
) -> Sequence[float]:
    return [scorer(pred, example.target_text) for pred, example in zip(predictions, examples)]
