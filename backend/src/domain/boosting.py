"""Gradient-boosted quantiles with an exact pinball line search per terminal leaf."""

from collections.abc import Sequence
from dataclasses import dataclass

from .checkpoint import fit_quantile_resumable
from .settings import FEATURES
from .trees import Tree, Vector

QUANTILES = (50, 90)


def shot_features(row: dict) -> Vector:
    return tuple(row[field] for field in FEATURES)


def empirical_quantile(values: Sequence[float], percentile: int) -> float:
    """Return the smallest value whose empirical CDF reaches the given percentile."""
    ordered = sorted(values)
    rank = (len(ordered) * percentile + 99) // 100
    return ordered[rank - 1]


@dataclass(frozen=True)
class QuantileModel:
    constant: float
    rate: float
    trees: tuple[Tree, ...]

    def predict(self, features: Vector) -> float:
        prediction = self.constant
        for tree in self.trees:
            prediction += self.rate * tree.predict(features)
        return prediction


def fit_quantile(features: Sequence[Vector], actual: Sequence[float], percentile: int,
                 settings: dict) -> QuantileModel:
    trees, _fitted, constant = fit_quantile_resumable(features, actual, percentile, settings, run_id=0)
    rate = settings["learning_rate"]
    return QuantileModel(constant, rate, tuple(trees))


def fit_quantile_with_sink(features: Sequence[Vector], actual: Sequence[float], percentile: int,
                           settings: dict, run_id: int, sink) -> QuantileModel:
    trees, _fitted, constant = fit_quantile_resumable(
        features, actual, percentile, settings, run_id=run_id, sink=sink)
    return QuantileModel(constant, settings["learning_rate"], tuple(trees))


def predict_shot(row: dict, models: tuple[QuantileModel, QuantileModel]) -> dict:
    vector = shot_features(row)
    raw_p50, raw_p90 = (model.predict(vector) for model in models)
    p50 = round(max(0.0, raw_p50), 6)
    p90 = round(max(0.0, raw_p50, raw_p90), 6)
    return {**row, "p50_minutes": p50, "p90_minutes": p90, "crossing_adjusted": raw_p90 < raw_p50}
