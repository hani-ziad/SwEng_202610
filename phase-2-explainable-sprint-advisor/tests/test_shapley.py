"""The Shapley explainer must be exact: closed form for linear models, and the
efficiency property (attributions sum to prediction minus background mean)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from phase2.shapley import exact_group_shapley


def _toy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
            "d": rng.normal(size=n),
        }
    )


def test_linear_model_matches_closed_form():
    X = _toy()
    y = 2.0 * X["a"] - 3.0 * X["b"] + 0.5 * X["c"] + 0.0 * X["d"] + 1.0
    lin = LinearRegression().fit(X, y)
    bg = X.iloc[:60]
    rows = X.iloc[60:70]
    groups = {"a": ["a"], "b": ["b"], "c": ["c"], "d": ["d"]}
    phi = exact_group_shapley(lambda f: lin.predict(f[list(X.columns)]), rows, bg, groups)
    for g in groups:
        expected = lin.coef_[list(X.columns).index(g)] * (rows[g] - bg[g].mean())
        assert np.allclose(phi[g].to_numpy(), expected.to_numpy(), atol=1e-8)


def test_efficiency_for_nonlinear_model_with_grouped_features():
    X = _toy()
    y = X["a"] * X["b"] + (X["c"] > 0) * 2.0 + 0.1 * X["d"]
    rf = RandomForestRegressor(n_estimators=40, max_depth=4, random_state=0).fit(X, y)
    bg = X.iloc[:30]
    rows = X.iloc[30:36]
    groups = {"ab": ["a", "b"], "c": ["c"], "d": ["d"]}
    predict = lambda f: rf.predict(f[list(X.columns)])
    phi = exact_group_shapley(predict, rows, bg, groups)
    total = phi[list(groups)].sum(axis=1) + phi["base_value"]
    assert np.allclose(total.to_numpy(), predict(rows), atol=1e-8)


def test_irrelevant_group_gets_zero_attribution():
    X = _toy()
    y = 4.0 * X["a"]
    lin = LinearRegression().fit(X, y)
    phi = exact_group_shapley(
        lambda f: lin.predict(f[list(X.columns)]),
        X.iloc[50:55],
        X.iloc[:50],
        {"a": ["a"], "rest": ["b", "c", "d"]},
    )
    assert np.allclose(phi["rest"].to_numpy(), 0.0, atol=1e-8)
