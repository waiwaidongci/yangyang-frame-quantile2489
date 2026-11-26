import json
import logging
import uuid

from sqlalchemy import delete, select

from src.config.database import Session
from src.constants.workflow import RunStatus
from src.domain.checkpoint import TrainingFenced
from src.domain.project import analyze
from src.models.checkpoint import RunCheckpoint
from src.models.dataset import Dataset
from src.models.run import Run
from src.services.audit import record
from src.services.progress import DatabaseProgress


def recover_jobs():
    """中断的运行重新入队并轮换围栏令牌，使任何旧进程的后续写入失效。"""
    with Session.begin() as session:
        runs = session.scalars(select(Run).where(Run.status == RunStatus.RUNNING)).all()
        for run in runs:
            run.status = RunStatus.QUEUED
            run.claim_token = None
            record(session, 'run.requeued', run.id)


def claim():
    with Session.begin() as session:
        run = session.scalar(select(Run).where(Run.status == RunStatus.QUEUED)
                             .order_by(Run.id).limit(1).with_for_update(skip_locked=True))
        if run is None:
            return None
        token = uuid.uuid4().hex
        run.status = RunStatus.RUNNING
        run.claim_token = token
        dataset = session.get(Dataset, run.dataset_id)
        record(session, 'run.started', run.id)
        return run.id, dataset.rows, run.settings, token


def execute(job):
    run_id, rows, settings, token = job
    result = None
    error_message = None
    progress = DatabaseProgress(run_id, token)
    try:
        result = analyze(rows, settings, run_id=run_id, sink=progress)
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except TrainingFenced:
        # 旧进程在重启后复活：接手进程拥有新令牌，这里直接放弃，不覆盖任何状态。
        logging.getLogger(__name__).warning('运行 %s 的旧进程围栏失效，已放弃提交。', run_id)
        return
    except ValueError as error:
        result = None
        error_message = str(error)
    except Exception:
        logging.getLogger(__name__).exception('计算运行 %s 失败', run_id)
        result = None
        error_message = '计算遇到内部错误，请检查台账后重试。'

    with Session.begin() as session:
        run = session.scalar(
            select(Run).where(Run.id == run_id, Run.claim_token == token,
                              Run.status == RunStatus.RUNNING).with_for_update()
        )
        if run is None:
            logging.getLogger(__name__).warning('运行 %s 已被接手进程接管，本次提交被拒绝。', run_id)
            return
        run.status = RunStatus.FAILED if error_message is not None else RunStatus.COMPLETED
        run.result = result
        run.error = error_message
        if run.status == RunStatus.COMPLETED:
            # 结果与一口气跑完一致后检查点不再需要，避免泄露中途轮次。
            session.execute(delete(RunCheckpoint).where(RunCheckpoint.run_id == run_id))
            run.claim_token = None
        record(session, 'run.' + run.status, run_id)
