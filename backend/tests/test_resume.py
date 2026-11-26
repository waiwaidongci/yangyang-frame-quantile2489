"""断点续跑与围栏的本地验证。

运行（在 backend 目录下）：
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 tests/test_resume.py
"""

import json
import os
import sys
import tempfile

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)

from src.domain.checkpoint import (  # noqa: E402
    TrainingFenced,
    fit_quantile_resumable,
    tree_to_data,
)
from src.domain.boosting import fit_quantile  # noqa: E402
from src.domain.fixtures import demo_rows  # noqa: E402
from src.domain.project import analyze, SPEC  # noqa: E402
from src.domain.settings import parse_settings  # noqa: E402

ROWS = demo_rows()
SETTINGS = parse_settings({})
OPTIONS = {**SETTINGS, "estimators": 12}  # 边界测试用小轮数加快速度


def train_rows():
    from src.domain.project import normalize_rows, shot_features

    records = sorted(normalize_rows(ROWS), key=lambda row: (row["scene_id"], row["id"]))
    train = [row for row in records if row["split"] == "train"]
    return [shot_features(row) for row in train], [row["actual_minutes"] for row in train]


class MemorySink:
    def __init__(self, rounds=None, reject_from=None):
        self.rounds = rounds if rounds is not None else []
        self.reject_from = reject_from

    def save_round(self, run_id, percentile, round_index, tree, fitted):
        if self.reject_from is not None and round_index >= self.reject_from:
            return False
        self.rounds.append({
            "round_index": round_index,
            "tree": tree,  # fit_quantile_resumable 已将树序列化为字典。
            "fitted": list(fitted),
        })
        return True

    def load_rounds(self, run_id, percentile):
        return list(self.rounds)


def model_signature(model):
    return [tree_to_data(tree) for tree in model.trees]


def test_full_run_equivalence():
    features, actual = train_rows()
    for percentile in (50, 90):
        baseline = fit_quantile(features, actual, percentile, OPTIONS)
        sink = MemorySink()
        trees, _fitted, constant = fit_quantile_resumable(
            features, actual, percentile, OPTIONS, run_id=1, sink=sink)
        assert constant == baseline.constant
        assert [tree_to_data(tree) for tree in trees] == model_signature(baseline)
        assert len(sink.rounds) == OPTIONS["estimators"]
        assert [item["round_index"] for item in sink.rounds] == list(range(OPTIONS["estimators"]))
    print("ok  一口气训练与逐轮落盘的树完全一致")


def test_crash_mid_round_resumes():
    features, actual = train_rows()
    crash_at = 7  # 第 7 轮写到一半进程消失，该轮不产生检查点
    for percentile in (50, 90):
        sink = MemorySink(reject_from=None)
        original_save = sink.save_round

        def crashing_save(run_id, q, round_index, tree, fitted):
            if round_index == crash_at:
                raise RuntimeError("simulated restart during write")
            return original_save(run_id, q, round_index, tree, fitted)

        sink.save_round = crashing_save
        try:
            fit_quantile_resumable(features, actual, percentile, OPTIONS, run_id=2, sink=sink)
        except RuntimeError:
            pass
        # 只有完整轮次落盘，崩溃轮次不存在。
        assert [item["round_index"] for item in sink.rounds] == list(range(crash_at))

        # 新进程从最后完整轮次续跑。
        resume_sink = MemorySink(rounds=list(sink.rounds))
        trees, _fitted, constant = fit_quantile_resumable(
            features, actual, percentile, OPTIONS, run_id=2, sink=resume_sink)
        baseline = fit_quantile(features, actual, percentile, OPTIONS)
        assert constant == baseline.constant
        assert [tree_to_data(tree) for tree in trees] == model_signature(baseline)
        assert [item["round_index"] for item in resume_sink.rounds] == list(range(OPTIONS["estimators"]))
        # 每个检查点的训练预测等于同轮模型在训练集上的逐步预测。
        for index, item in enumerate(resume_sink.rounds):
            previous = [baseline.constant] * len(features) if index == 0 \
                else resume_sink.rounds[index - 1]["fitted"]
            step_tree = baseline.trees[index]
            rebuilt_values = [
                prev + baseline.rate * step_tree.predict(vector)
                for prev, vector in zip(previous, features)
            ]
            assert item["fitted"] == rebuilt_values
    print("ok  写到一半的轮次被丢弃，续跑结果与一口气训练逐项相同")


def test_fence_revokes_old_worker():
    features, actual = train_rows()
    percentile = 50
    sink = MemorySink()
    try:
        fit_quantile_resumable(features, actual, percentile, OPTIONS, run_id=3,
                               sink=MemorySink(reject_from=5))
    except TrainingFenced as error:
        assert "围栏" in str(error)
    else:
        raise AssertionError("围栏失效后应当抛出 TrainingFenced")

    # 接手进程拿着前 5 个完整轮次继续，结果仍与一口气训练一致。
    accepted = MemorySink()
    fit_quantile_resumable(features, actual, percentile,
                           {**OPTIONS, "estimators": 5}, run_id=3, sink=accepted)
    handoff = MemorySink(rounds=list(accepted.rounds))
    trees, _fitted, constant = fit_quantile_resumable(
        features, actual, percentile, OPTIONS, run_id=3, sink=handoff)
    baseline = fit_quantile(features, actual, percentile, OPTIONS)
    assert [tree_to_data(tree) for tree in trees] == model_signature(baseline)
    print("ok  旧进程写入被围栏拒绝并收到中断信号，接手进程进度唯一有效")


def build_sqlite_workspace():
    tmp_dir = tempfile.mkdtemp(prefix="resume-test-")
    db_path = os.path.join(tmp_dir, "work.db")
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    os.environ["REDIS_URL"] = "redis://127.0.0.1:6399/0"
    os.environ["ARTIFACT_DIR"] = tmp_dir
    # 延迟导入，确保 DATABASE_URL 生效。
    for name in list(sys.modules):
        if name.startswith("src.") or name.startswith("migrations."):
            del sys.modules[name]
    from migrations.bootstrap import bootstrap
    from src.config.database import Session
    from src.constants.workflow import RunStatus, Role
    from src.models.account import Account
    from src.models.dataset import Dataset
    from src.models.run import Run
    from src.services import execution
    from src.services.progress import DatabaseProgress

    bootstrap()
    with Session.begin() as session:
        account = Account(email="analyst@demo.local", role=Role.ANALYST, password_hash="x")
        session.add(account)
        session.flush()
        dataset = Dataset(name=SPEC["dataset_name"], rows=ROWS, checksum="0" * 64,
                          created_by=account.id)
        session.add(dataset)
        session.flush()
        run = Run(label="crash-run", dataset_id=dataset.id, created_by=account.id,
                  created_at="2026-11-26T00:00:00+00:00", status=RunStatus.QUEUED,
                  settings={}, result=None, error=None)
        session.add(run)
        session.flush()
        run_id = run.id
    return tmp_dir, run_id, execution, DatabaseProgress, RunStatus, Session


def test_database_resume_and_fence():
    tmp_dir, run_id, execution, DatabaseProgress, RunStatus, Session = build_sqlite_workspace()
    from src.domain.checkpoint import fit_quantile_resumable
    from src.models.checkpoint import RunCheckpoint
    from src.models.run import Run
    from sqlalchemy import func, select

    baseline = analyze(ROWS, {})

    # 第一个 worker 领取并在 P50 第 9 轮写入前崩溃。
    job_a = execution.claim()
    assert job_a is not None and job_a[0] == run_id
    _id, _rows, _settings, token_a = job_a
    sink_a = DatabaseProgress(run_id, token_a)
    features, actual = train_rows()
    original = sink_a.save_round

    def save_then_crash(q_run, percentile, round_index, tree, fitted):
        if percentile == 50 and round_index == 9:
            raise RuntimeError("simulated restart during write")
        return original(q_run, percentile, round_index, tree, fitted)

    sink_a.save_round = save_then_crash
    try:
        fit_quantile_resumable(features, actual, 50, SETTINGS, run_id, sink=sink_a)
    except RuntimeError:
        pass
    sink_a.save_round = original  # 还原：进程已重启，后续调用模拟旧进程的迟到写入。
    with Session() as session:
        saved = session.scalars(select(RunCheckpoint).where(
            RunCheckpoint.run_id == run_id, RunCheckpoint.percentile == 50)
            .order_by(RunCheckpoint.round_index)).all()
        assert [item.round_index for item in saved] == list(range(9))
        assert session.get(Run, run_id).status == RunStatus.RUNNING

    # 重启：旧运行重新入队、围栏令牌清空。
    execution.recover_jobs()
    with Session() as session:
        run = session.get(Run, run_id)
        assert run.status == RunStatus.QUEUED and run.claim_token is None

    # 旧进程复活：令牌已失效，写入必须被拒绝。
    assert sink_a.save_round(run_id, 50, 9, {"kind": "leaf", "correction": 1.0}, [1.0]) is False

    # 接手进程领取并续跑完整运行。
    job_b = execution.claim()
    assert job_b[0] == run_id
    token_b = job_b[3]
    assert token_b != token_a
    execution.execute(job_b)
    with Session() as session:
        run = session.get(Run, run_id)
        assert run.status == RunStatus.COMPLETED
        assert run.claim_token is None
        assert run.error is None
        assert json.dumps(run.result, sort_keys=True, ensure_ascii=False) == \
            json.dumps(baseline, sort_keys=True, ensure_ascii=False)
        # 完成后中途检查点已清理。
        assert session.scalar(select(func.count()).select_from(RunCheckpoint)) == 0
    metrics = baseline["metrics"]
    assert metrics["P50 留出 Pinball"] == 2.95042
    assert metrics["P90 留出 Pinball"] == 1.996761
    assert metrics["P90 留出覆盖率"] == 0.828571
    assert metrics["本班已预留分钟"] == 166

    # 旧进程的迟到提交必须被拒绝，运行结果不被覆盖。
    execution.execute((run_id, ROWS, {}, token_a))
    with Session() as session:
        run = session.get(Run, run_id)
        assert run.status == RunStatus.COMPLETED
        assert json.dumps(run.result, sort_keys=True, ensure_ascii=False) == \
            json.dumps(baseline, sort_keys=True, ensure_ascii=False)
    print("ok  SQLite 下崩溃续跑、围栏拒绝旧进程，结果与一口气跑完逐项相同")


def test_bootstrap_upgrades_legacy_table():
    tmp_dir = tempfile.mkdtemp(prefix="legacy-test-")
    db_path = os.path.join(tmp_dir, "legacy.db")
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    for name in list(sys.modules):
        if name.startswith("src.") or name.startswith("migrations."):
            del sys.modules[name]
    from sqlalchemy import inspect, text

    from src.config.database import engine
    with engine.begin() as connection:
        connection.execute(text(
            "CREATE TABLE analysis_runs ("
            "id INTEGER PRIMARY KEY, label VARCHAR(140), dataset_id INTEGER, "
            "created_by INTEGER, created_at VARCHAR(32), status VARCHAR(16), "
            "settings JSON, result JSON, error TEXT)"
        ))
    from migrations.bootstrap import bootstrap

    bootstrap()
    columns = {column["name"] for column in inspect(engine).get_columns("analysis_runs")}
    assert "claim_token" in columns
    tables = inspect(engine).get_table_names()
    assert "run_checkpoints" in tables
    print("ok  旧库启动时幂等补齐 claim_token 与检查点表")


def test_p90_crash_after_p50_finished():
    tmp_dir, run_id, execution, DatabaseProgress, RunStatus, Session = build_sqlite_workspace()
    from src.domain.checkpoint import fit_quantile_resumable
    from src.models.checkpoint import RunCheckpoint
    from src.models.run import Run
    from sqlalchemy import select

    job_a = execution.claim()
    token_a = job_a[3]
    sink_a = DatabaseProgress(run_id, token_a)
    features, actual = train_rows()
    original = sink_a.save_round

    def save_then_crash(q_run, percentile, round_index, tree, fitted):
        # P50 全部 60 轮完成后，P90 第 3 轮写到一半崩溃。
        if percentile == 90 and round_index == 3:
            raise RuntimeError("simulated restart during P90 write")
        return original(q_run, percentile, round_index, tree, fitted)

    sink_a.save_round = save_then_crash
    try:
        fit_quantile_resumable(features, actual, 50, SETTINGS, run_id, sink=sink_a)
        fit_quantile_resumable(features, actual, 90, SETTINGS, run_id, sink=sink_a)
    except RuntimeError:
        pass
    sink_a.save_round = original
    with Session() as session:
        p50_rounds = session.scalars(select(RunCheckpoint.round_index).where(
            RunCheckpoint.run_id == run_id, RunCheckpoint.percentile == 50)).all()
        p90_rounds = session.scalars(select(RunCheckpoint.round_index).where(
            RunCheckpoint.run_id == run_id, RunCheckpoint.percentile == 90)).all()
        assert sorted(p50_rounds) == list(range(SETTINGS["estimators"]))
        assert sorted(p90_rounds) == list(range(3))

    execution.recover_jobs()
    job_b = execution.claim()
    execution.execute(job_b)
    baseline = analyze(ROWS, {})
    with Session() as session:
        run = session.get(Run, run_id)
        assert run.status == RunStatus.COMPLETED
        assert json.dumps(run.result, sort_keys=True, ensure_ascii=False) == \
            json.dumps(baseline, sort_keys=True, ensure_ascii=False)
    print("ok  P50 完成后 P90 中途崩溃，续跑沿用两套检查点且结果逐项相同")


def test_uninterrupted_worker_path():
    tmp_dir, run_id, execution, _progress, RunStatus, Session = build_sqlite_workspace()
    from src.models.checkpoint import RunCheckpoint
    from src.models.run import Run
    from sqlalchemy import func, select

    job = execution.claim()
    assert job[3] is not None
    execution.execute(job)
    baseline = analyze(ROWS, {})
    with Session() as session:
        run = session.get(Run, run_id)
        assert run.status == RunStatus.COMPLETED and run.claim_token is None
        assert json.dumps(run.result, sort_keys=True, ensure_ascii=False) == \
            json.dumps(baseline, sort_keys=True, ensure_ascii=False)
        assert session.scalar(select(func.count()).select_from(RunCheckpoint)) == 0

    # 没有中断运行时 recover 是空操作。
    execution.recover_jobs()
    with Session() as session:
        assert session.get(Run, run_id).status == RunStatus.COMPLETED
    print("ok  无中断时 worker 正常完成且不残留检查点")


def test_http_workflow_with_restart():
    tmp_dir = tempfile.mkdtemp(prefix="http-test-")
    os.environ["DATABASE_URL"] = f"sqlite:///{os.path.join(tmp_dir, 'http.db')}"
    os.environ["REDIS_URL"] = "redis://127.0.0.1:6399/0"
    os.environ["ARTIFACT_DIR"] = tmp_dir
    os.environ["SEED_DEMO"] = "false"
    os.environ["JWT_SECRET"] = "test-secret"
    for name in list(sys.modules):
        if name.startswith("src.") or name.startswith("migrations."):
            del sys.modules[name]
    from fastapi.testclient import TestClient

    from migrations.bootstrap import bootstrap
    from src.config.database import Session
    from src.constants.workflow import Role
    from src.main import app
    from src.models.account import Account
    from src.models.dataset import Dataset
    from src.services import execution
    from src.services.progress import DatabaseProgress
    from src.utils.security import hash_password
    from src.domain.checkpoint import fit_quantile_resumable

    bootstrap()
    with Session.begin() as session:
        analyst = Account(email="analyst@demo.local", role=Role.ANALYST,
                          password_hash=hash_password("Demo-2611!"))
        admin = Account(email="admin@demo.local", role=Role.ADMIN,
                        password_hash=hash_password("Demo-2611!"))
        session.add_all([analyst, admin])
        session.flush()
        session.add(Dataset(name=SPEC["dataset_name"], rows=ROWS, checksum="0" * 64,
                            created_by=analyst.id))
    client = TestClient(app)
    token = client.post("/api/auth/login",
                        json={"email": "analyst@demo.local", "password": "Demo-2611!"}
                        ).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    created = client.post("/api/runs", headers=headers,
                          json={"dataset_id": 1, "label": "restart-e2e", "settings": {}})
    assert created.status_code == 202
    run_id = created.json()["id"]

    # worker A 领取后在 P50 第 20 轮崩溃。
    job_a = execution.claim()
    token_a = job_a[3]
    sink_a = DatabaseProgress(run_id, token_a)
    features, actual = train_rows()
    original = sink_a.save_round

    def save_then_crash(q_run, percentile, round_index, tree, fitted):
        if percentile == 50 and round_index == 20:
            raise RuntimeError("simulated restart")
        return original(q_run, percentile, round_index, tree, fitted)

    sink_a.save_round = save_then_crash
    try:
        fit_quantile_resumable(features, actual, 50, SETTINGS, run_id, sink=sink_a)
    except RuntimeError:
        pass

    # 重启 worker：recover 后由接手进程完整跑完。
    execution.recover_jobs()
    job_b = execution.claim()
    execution.execute(job_b)

    detail = client.get(f"/api/runs/{run_id}", headers=headers).json()
    assert detail["status"] == "completed"
    baseline = analyze(ROWS, {})
    assert detail["result"] == baseline
    assert detail["result"]["metrics"]["P50 留出 Pinball"] == 2.95042

    # 审批通过前导出被拒，通过后可下载。
    admin_token = client.post("/api/auth/login",
                              json={"email": "admin@demo.local", "password": "Demo-2611!"}
                              ).json()["access_token"]
    forbidden = client.get(f"/api/runs/{run_id}/export/shift_queue", headers=headers)
    assert forbidden.status_code == 409
    review = client.post(f"/api/runs/{run_id}/review",
                         headers={"Authorization": f"Bearer {admin_token}"},
                         json={"decision": "approved", "note": "核对顺延镜头"})
    assert review.status_code == 201
    exported = client.get(f"/api/runs/{run_id}/export/shift_queue", headers=headers)
    assert exported.status_code == 200 and "text/csv" in exported.headers["content-type"]
    assert exported.text.count("\n") == 13  # 表头 + 12 个待排镜头
    print("ok  HTTP 工作流：重启续跑后结果可查，审批与导出行为不变")


if __name__ == "__main__":
    test_full_run_equivalence()
    test_crash_mid_round_resumes()
    test_fence_revokes_old_worker()
    test_database_resume_and_fence()
    test_p90_crash_after_p50_finished()
    test_uninterrupted_worker_path()
    test_bootstrap_upgrades_legacy_table()
    test_http_workflow_with_restart()
    print("全部断点续跑测试通过。")
