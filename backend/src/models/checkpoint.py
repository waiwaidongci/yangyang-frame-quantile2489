from sqlalchemy import JSON, CheckConstraint, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from src.config.database import Base


class RunCheckpoint(Base):
    """每个分位、每个完整轮次一行：该轮新树与更新后的训练预测。"""

    __tablename__ = 'run_checkpoints'
    __table_args__ = (
        UniqueConstraint('run_id', 'percentile', 'round_index', name='uq_run_checkpoint_round'),
        CheckConstraint('percentile IN (50, 90)', name='ck_run_checkpoint_percentile'),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey('analysis_runs.id', ondelete='CASCADE'), index=True)
    percentile: Mapped[int] = mapped_column(Integer)
    round_index: Mapped[int] = mapped_column(Integer)
    tree: Mapped[dict] = mapped_column(JSON)
    fitted: Mapped[list] = mapped_column(JSON)
