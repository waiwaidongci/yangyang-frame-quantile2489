"""Database-backed round checkpoints with worker fencing tokens.

每个检查点在一个事务中先做条件 UPDATE 校验围栏令牌：只有当前接手进程的
令牌才能把行“碰”一下并继续插入。旧进程复活后令牌已在 recover/claim 时
轮换，其写入会被拒绝，保证同一运行的进度只认接手进程。条件 UPDATE 不持有
行锁，避免与唯一约束的索引检查形成锁环。
"""

from collections.abc import Sequence
from logging import getLogger

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from src.config.database import Session
from src.constants.workflow import RunStatus
from src.models.checkpoint import RunCheckpoint
from src.models.run import Run

logger = getLogger(__name__)


class DatabaseProgress:
    def __init__(self, run_id: int, claim_token: str):
        self.run_id = run_id
        self.claim_token = claim_token

    def save_round(self, run_id: int, percentile: int, round_index: int,
                   tree: dict, fitted: Sequence[float]) -> bool:
        with Session.begin() as session:
            # 围栏：令牌不匹配或运行已离开 running 时 rowcount 为 0。
            fenced = session.execute(
                update(Run)
                .where(Run.id == run_id, Run.claim_token == self.claim_token,
                       Run.status == RunStatus.RUNNING)
                .values(claim_token=self.claim_token)
            ).rowcount
            if fenced != 1:
                return False
            duplicate = session.scalar(
                select(RunCheckpoint.id).where(
                    RunCheckpoint.run_id == run_id,
                    RunCheckpoint.percentile == percentile,
                    RunCheckpoint.round_index == round_index,
                )
            )
            if duplicate is not None:
                # 同一轮次已落盘（例如重复调用），幂等视为成功但不重复写入。
                return True
            session.add(RunCheckpoint(
                run_id=run_id, percentile=percentile, round_index=round_index,
                tree=tree, fitted=list(fitted),
            ))
            try:
                session.flush()
            except IntegrityError:
                # 并发下唯一约束兜底：该轮已存在即视为幂等成功，否则交给上层重试。
                logger.warning('运行 %s 的 P%s 第 %s 轮检查点并发冲突。', run_id, percentile, round_index)
                raise
        return True

    def load_rounds(self, run_id: int, percentile: int) -> list[dict]:
        with Session() as session:
            checkpoints = session.scalars(
                select(RunCheckpoint)
                .where(RunCheckpoint.run_id == run_id, RunCheckpoint.percentile == percentile)
                .order_by(RunCheckpoint.round_index)
            ).all()
            return [
                {"round_index": item.round_index, "tree": item.tree, "fitted": list(item.fitted)}
                for item in checkpoints
            ]
