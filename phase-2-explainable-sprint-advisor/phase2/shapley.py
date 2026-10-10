"""Exact group-level Shapley attributions (no ``shap`` package needed).

Interventional Shapley values with a background sample, computed exactly by
enumerating every coalition of feature *groups*. With a handful of groups
(scope, sprint length, recent history, capacity, project) that is only 2^G
coalitions, so no sampling approximation is involved.

value(S) for one explained row x:
    the model's average prediction when the features in coalition S are set to
    x's values and every other feature is drawn from the background rows.

The attributions satisfy the efficiency property:
    sum_g phi_g  =  f(x) - E_background[f]
"""
from __future__ import annotations

from itertools import combinations
from math import factorial
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd


def _coalition_masks(n_groups: int):
    for r in range(n_groups + 1):
        for subset in combinations(range(n_groups), r):
            yield frozenset(subset)


def exact_group_shapley(
    predict: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    background: pd.DataFrame,
    groups: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Return a DataFrame (rows of X x groups) of Shapley values.

    ``predict`` maps a frame with the same columns as X to a 1-D array.
    A ``base_value`` column holds E_background[f] so that
    base_value + sum(groups) == f(x) for every row.
    """
    names = list(groups)
    G = len(names)
    cols = list(X.columns)
    background = background[cols].reset_index(drop=True)
    nb = len(background)
    masks = list(_coalition_masks(G))
    mask_index = {m: i for i, m in enumerate(masks)}

    weights = {}
    for s in range(G):
        weights[s] = factorial(s) * factorial(G - s - 1) / factorial(G)

    base_value = float(np.mean(predict(background)))
    out = np.zeros((len(X), G))
    for ri, (_, x) in enumerate(X.iterrows()):
        frames = []
        for m in masks:
            replaced = [c for gi in m for c in groups[names[gi]]]
            df = background.copy()
            for c in replaced:
                df[c] = x[c]
            frames.append(df)
        preds = np.asarray(predict(pd.concat(frames, ignore_index=True)), dtype=float)
        v = preds.reshape(len(masks), nb).mean(axis=1)
        for gi in range(G):
            phi = 0.0
            others = [j for j in range(G) if j != gi]
            for r in range(G):
                for subset in combinations(others, r):
                    s = frozenset(subset)
                    phi += weights[r] * (v[mask_index[s | {gi}]] - v[mask_index[s]])
            out[ri, gi] = phi
    res = pd.DataFrame(out, columns=names, index=X.index)
    res["base_value"] = base_value
    return res
