"""Round-level checkpoints for resumable quantile boosting.

每个完整轮次一次性落盘该轮新树与更新后的训练预测。写到一半的轮次不产生
检查点，重启后只会从最后一个完整轮次之后继续。旧进程即使存活，其围栏令牌
失效后写入会被拒绝，并以 :class:`TrainingFenced` 通知调用方停止计算。
"""

from collections.abc import Sequence
from typing import Protocol

from .trees import Branch, Leaf, Tree, Vector


class TrainingFenced(RuntimeError):
    """当前进程已不再是该运行的接手进程，必须立刻放弃后续轮次。"""


def tree_to_data(tree: Tree) -> dict:
    if isinstance(tree, Leaf):
        return {"kind": "leaf", "correction": tree.correction}
    return {
        "kind": "branch",
        "feature": tree.feature,
        "threshold": tree.threshold,
        "left": tree_to_data(tree.left),
        "right": tree_to_data(tree.right),
    }


def tree_from_data(data: dict) -> Tree:
    if data["kind"] == "leaf":
        return Leaf(data["correction"])
    return Branch(
        data["feature"],
        data["threshold"],
        tree_from_data(data["left"]),
        tree_from_data(data["right"]),
    )


class ProgressSink(Protocol):
    def save_round(self, run_id: int, percentile: int, round_index: int,
                   tree: dict, fitted: Sequence[float]) -> bool:
        """原子保存一个完整轮次。返回 False 表示围栏失效，调用方应停止。"""

    def load_rounds(self, run_id: int, percentile: int) -> list[dict]:
        """返回按轮次升序的已完整落盘检查点。"""


def fit_quantile_resumable(features: Sequence[Vector], actual: Sequence[float], percentile: int,
                           settings: dict, run_id: int,
                           sink: ProgressSink | None = None) -> tuple[list[Tree], list[float], float]:
    """逐轮训练提升树，并在每个完整轮次后通过 sink 落盘。

    返回 ``(trees, fitted, constant)``。sink 中已有检查点时从最后一个完整
    轮次续跑，恢复后的树与预测和一口气训练完全一致。
    """
    from .boosting import empirical_quantile
    from .trees import fit_tree

    constant = empirical_quantile(actual, percentile)
    rate = settings["learning_rate"]
    alpha = percentile / 100
    total_rounds = settings["estimators"]

    saved = sink.load_rounds(run_id, percentile) if sink is not None else []
    if saved:
        start_round = saved[-1]["round_index"] + 1
        trees = [tree_from_data(item["tree"]) for item in saved]
        fitted = [float(value) for value in saved[-1]["fitted"]]
    else:
        start_round = 0
        trees = []
        fitted = [constant] * len(actual)

    for round_index in range(start_round, total_rounds):
        residuals = [observed - estimate for observed, estimate in zip(actual, fitted)]
        # Zero is a valid subgradient at an exact hit.
        gradients = [alpha if residual > 0 else alpha - 1 if residual < 0 else 0.0 for residual in residuals]

        def correction(members: list[int]) -> float:
            return empirical_quantile([residuals[index] for index in members], percentile)

        tree = fit_tree(features, gradients, correction, settings["max_depth"], settings["min_leaf"])
        next_fitted = [estimate + rate * tree.predict(vector)
                       for estimate, vector in zip(fitted, features)]
        if sink is not None:
            accepted = sink.save_round(run_id, percentile, round_index,
                                       tree_to_data(tree), next_fitted)
            if not accepted:
                raise TrainingFenced(f"运行 {run_id} 的 P{percentile} 训练围栏已失效。")
        trees.append(tree)
        fitted = next_fitted
    return trees, fitted, constant
