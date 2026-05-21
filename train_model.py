import numpy as np
import polars as pl
from typing import Any

from config import *


def train_and_estimate(X_train, y_train, X_test, y_test, model, metric_func, params: dict) -> float | Any | np.ndarray:
    """Обучение и оценка."""

    model = model(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metric = metric_func(y_test, y_pred)
    
    return metric


if __name__ == "__main__":
    pl.set_random_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    X_train = pl.read_csv(X_TRAIN_PATH)
    y_train = pl.read_csv(Y_TRAIN_PATH)
    X_test = pl.read_csv(X_TEST_PATH)
    y_test = pl.read_csv(Y_TEST_PATH)

    metric_value = train_and_estimate(X_train, y_train, X_test, y_test, MODEL, METRIC_FUNC, params={})
    print(f"model: {MODEL.__name__}, metric: {METRIC_FUNC.__name__} {metric_value}")