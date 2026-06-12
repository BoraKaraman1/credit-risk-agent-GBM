"""Database logging helpers for monitoring agents."""

import json

from sqlalchemy import text


DRIFT_LOG_INSERT_SQL = text("""
    INSERT INTO drift_log (metric_name, metric_value, model_version, details)
    VALUES (:metric_name, :metric_value, :model_version, CAST(:details AS jsonb))
""")


def insert_drift_log(conn, *, metric_name: str, metric_value: float,
                     model_version: str, details: dict) -> None:
    """Insert one monitoring metric audit record."""
    conn.execute(
        DRIFT_LOG_INSERT_SQL,
        {
            "metric_name": metric_name,
            "metric_value": metric_value,
            "model_version": model_version,
            "details": json.dumps(details),
        },
    )
