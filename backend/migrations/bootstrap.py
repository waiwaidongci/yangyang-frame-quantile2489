from sqlalchemy import inspect, text

from src.config.database import Base, engine
from src.models import account, audit, checkpoint, dataset, review, run  # noqa: F401


def bootstrap():
    Base.metadata.create_all(engine)
    inspector = inspect(engine)
    columns = {column['name'] for column in inspector.get_columns('analysis_runs')}
    if 'claim_token' not in columns:
        with engine.begin() as connection:
            connection.execute(text('ALTER TABLE analysis_runs ADD COLUMN claim_token VARCHAR(64)'))
