"""
Models: Elastic Net and Random Forest with Expanding Window
=============================================================

This module implements the two core ML models from our simplified Kelly paper
replication: elastic net (penalized linear regression) and random forest
(tree-based ensemble). The Kelly paper tests these plus gradient-boosted trees,
neural networks, and PCA regression. We start with the two that are most
interpretable and easiest to debug.

The most important function here is `expanding_window_predict`. This is the
function that prevents lookahead bias — the single most common mistake in
backtesting. If you get this wrong, your entire analysis is meaningless because
you're using future data to make "predictions."

On a risk desk, you'd call this "walk-forward analysis" or "out-of-sample
testing." It's the difference between a backtest that works on paper and a
strategy that works in production.
"""

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. ELASTIC NET
# ---------------------------------------------------------------------------
# Elastic net = linear regression with L1 (lasso) + L2 (ridge) penalties.
# The Kelly paper uses it as the baseline linear model. It's the natural
# first model to try because:
#   - It handles multicollinearity (correlated features like mom_6m and mom_12m)
#   - L1 penalty does feature selection (sets weak coefficients to zero)
#   - It's fast to train, easy to interpret
#
# On your risk desk, think of this as a constrained regression where you
# penalize the model for using too many factors or giving any factor too
# much weight. It's the quantitative version of "don't overfit."

def train_elastic_net(
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
) -> ElasticNetCV:
    """
    Train an elastic net model with cross-validated regularization.

    ElasticNetCV automatically selects the best alpha (regularization strength)
    and l1_ratio (lasso vs. ridge mix) via time-series-aware cross-validation.
    """
    # Standardize features — elastic net is sensitive to scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
        cv=cv,
        max_iter=10000,
        n_jobs=-1,
    )
    model.fit(X_scaled, y)

    # Store the scaler on the model so we can transform at prediction time
    model.scaler_ = scaler

    return model


def predict_elastic_net(model: ElasticNetCV, X: pd.DataFrame) -> np.ndarray:
    """Generate predictions from a fitted elastic net (applies stored scaler)."""
    X_scaled = model.scaler_.transform(X)
    return model.predict(X_scaled)


# ---------------------------------------------------------------------------
# 2. RANDOM FOREST
# ---------------------------------------------------------------------------
# Random forest = ensemble of decision trees, each trained on a random
# subset of the data and features. The Kelly paper finds that tree-based
# models outperform linear models for return prediction because they
# capture nonlinear interactions between features.
#
# Example of a nonlinear interaction the elastic net misses:
#   "Momentum works well for large-cap stocks but reverses for micro-caps"
# A tree can split on size first, then on momentum — capturing the
# interaction without you having to specify it.
#
# TODO: Consider adding hyperparameter tuning (n_estimators, max_depth,
#       min_samples_leaf) via cross-validation. For now, sensible defaults.

def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 100,
) -> RandomForestRegressor:
    """
    Train a random forest regressor.

    No need to standardize features — trees are scale-invariant. This is one
    advantage of tree-based models over linear models.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=6,          # Shallow trees to avoid overfitting
        min_samples_leaf=20,  # Require decent sample in each leaf
        max_features="sqrt",  # Random feature subset at each split
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# 3. EXPANDING WINDOW PREDICTION
# ---------------------------------------------------------------------------
# This is the most important function in the project. If you get this wrong,
# your backtest is meaningless.
#
# The idea: at each month t, we train the model using ONLY data from months
# 1 through t-1, then predict returns at month t. We never use future data.
# As time progresses, the training window expands (more history = better
# estimates), but we never look ahead.
#
# This is different from a "rolling window" where you drop old data. The
# Kelly paper uses expanding windows — they find that more data generally
# helps, which makes sense for a cross-sectional model where the signal
# is in the relative ranking of stocks.
#
# On your risk desk: this is how you'd evaluate any model before putting
# real money behind it. Bloomberg's backtesting tools do the same thing
# under the hood. The fancy name is "walk-forward optimization."

def expanding_window_predict(
    data: pd.DataFrame,
    features: list[str],
    target: str,
    model_fn: Callable,
    predict_fn: Callable | None = None,
    min_periods: int = 36,
) -> pd.DataFrame:
    """
    Walk-forward prediction with expanding training window.

    At each time step t (where t >= min_periods):
        1. Train on all data from periods 0..t-1
        2. Predict for period t
        3. Move to t+1 and repeat

    This prevents lookahead bias — the model never sees future data.

    Parameters:
        data:        DataFrame with a 'date' column (or DatetimeIndex level)
                     plus feature columns and target column
        features:    List of feature column names
        target:      Name of the target column (e.g., 'next_month_return')
        model_fn:    Callable that takes (X, y) and returns a fitted model
        predict_fn:  Callable that takes (model, X) and returns predictions.
                     If None, uses model.predict(X).
        min_periods: Minimum number of months of training data before we
                     start predicting (default 36 = 3 years)

    Returns:
        DataFrame with columns ['date', 'ticker', 'prediction', 'actual']
    """
    # Get sorted unique dates
    if "date" in data.columns:
        dates = sorted(data["date"].unique())
    else:
        # Assume first level of MultiIndex is date
        dates = sorted(data.index.get_level_values(0).unique())

    if len(dates) < min_periods + 1:
        raise ValueError(
            f"Need at least {min_periods + 1} periods, got {len(dates)}. "
            f"Reduce min_periods or use more data."
        )

    all_predictions = []

    for t in range(min_periods, len(dates)):
        train_dates = dates[:t]
        test_date = dates[t]

        # Split data
        if "date" in data.columns:
            train_mask = data["date"].isin(train_dates)
            test_mask = data["date"] == test_date
        else:
            train_mask = data.index.get_level_values(0).isin(train_dates)
            test_mask = data.index.get_level_values(0) == test_date

        train_data = data.loc[train_mask]
        test_data = data.loc[test_mask]

        X_train = train_data[features].dropna()
        y_train = train_data.loc[X_train.index, target]

        X_test = test_data[features].dropna()
        if X_test.empty:
            continue

        y_test = test_data.loc[X_test.index, target]

        # Drop any remaining NaN in target
        valid_train = y_train.notna()
        X_train = X_train.loc[valid_train]
        y_train = y_train.loc[valid_train]

        if len(X_train) < 50:
            # Not enough training data this period — skip
            continue

        # Train and predict
        model = model_fn(X_train, y_train)

        if predict_fn is not None:
            preds = predict_fn(model, X_test)
        else:
            preds = model.predict(X_test)

        # Collect results
        result = pd.DataFrame({
            "prediction": preds,
            "actual": y_test.values,
        }, index=X_test.index)

        all_predictions.append(result)

    if not all_predictions:
        raise ValueError("No predictions generated. Check data and min_periods.")

    return pd.concat(all_predictions)


# ---------------------------------------------------------------------------
# 4. CONVENIENCE WRAPPERS
# ---------------------------------------------------------------------------
# These wrap the model functions into the signature that expanding_window_predict
# expects. Use these directly if you don't need custom hyperparameters.

# TODO: Add a gradient-boosted tree model (XGBoost or LightGBM). The Kelly
#       paper finds that boosted trees perform as well as neural networks
#       and are easier to tune. This would be a natural next step.

# TODO: Add a simple OLS baseline. Comparing ML models to plain OLS helps
#       you understand how much of the predictive power comes from
#       nonlinearity vs. feature selection vs. regularization.
