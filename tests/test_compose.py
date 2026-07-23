from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
COMPOSE = yaml.safe_load((ROOT / "docker-compose.yml").read_text())


def test_database_migration_gates_database_consumers():
    services = COMPOSE["services"]
    migration = services["migrate"]

    assert migration["restart"] == "no"
    assert migration["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert migration["environment"]["SCHEMA_MIGRATION_REVISION"]
    assert any(
        "supabase_schema.sql:/migrations/001_schema.sql:ro" in volume
        for volume in migration["volumes"]
    )
    assert migration["command"] == [
        "psql",
        "--set=ON_ERROR_STOP=1",
        "--file=/migrations/001_schema.sql",
    ]

    for service_name in ("api", "airflow"):
        dependency = services[service_name]["depends_on"]["migrate"]
        assert dependency["condition"] == "service_completed_successfully"
