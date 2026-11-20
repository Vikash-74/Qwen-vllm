import logging
import re
from datetime import datetime, timezone

import clickhouse_connect

logger = logging.getLogger(__name__)

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

COLUMNS = [
    "source_metric",
    "forecast_metric",
    "generated_at",
    "target_ts",
    "horizon",
    "forecast_value",
    "forecast_lower",
    "forecast_upper",
]

CREATE_DATABASE = "CREATE DATABASE IF NOT EXISTS {database}"

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS {database}.{table}
(
    source_metric   LowCardinality(String),
    forecast_metric LowCardinality(String),
    generated_at    DateTime64(3, 'UTC'),
    target_ts       DateTime('UTC'),
    horizon         UInt16,
    forecast_value  Float32,
    forecast_lower  Float32,
    forecast_upper  Float32
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(target_ts)
ORDER BY (forecast_metric, target_ts, generated_at)
TTL target_ts + INTERVAL {ttl_days} DAY
"""


def _check_identifier(value, what):
    if not _IDENTIFIER.match(value):
        raise ValueError(f"Invalid ClickHouse {what}: {value!r}")
    return value


class ClickHouseSink:
    """
    Stores every forecast step with the timestamp it refers to
    (target_ts), so Grafana can plot future values on a time axis.

    Every cycle inserts new rows (never updates). Each row records when
    the forecast was made (generated_at); queries pick the latest forecast
    per target_ts with argMax.
    """

    def __init__(
        self,
        host="localhost",
        port=8123,
        username="default",
        password="",
        database="monitoring",
        table="vllm_forecasts",
        secure=False,
        ttl_days=90,
    ):
        self.database = _check_identifier(database, "database")
        self.table_name = _check_identifier(table, "table")
        self.table = f"{self.database}.{self.table_name}"

        self.client = clickhouse_connect.get_client(
            host=host,
            port=port,
            username=username,
            password=password,
            secure=secure,
        )

        self.client.command(CREATE_DATABASE.format(database=self.database))
        self.client.command(
            CREATE_TABLE.format(
                database=self.database,
                table=self.table_name,
                ttl_days=int(ttl_days),
            )
        )

        logger.info("ClickHouse sink ready: %s", self.table)

    def write_forecast(self, source_metric, forecast_metric, target_timestamps, forecast):
        """
        target_timestamps: tz-aware pandas Timestamps, one per forecast step.
        forecast: dict with 'mean', 'lower', 'upper' arrays of equal length.
        """

        generated_at = datetime.now(timezone.utc)

        rows = []
        for index, target in enumerate(target_timestamps):
            rows.append(
                [
                    source_metric,
                    forecast_metric,
                    generated_at,
                    target.to_pydatetime(),
                    index + 1,
                    float(forecast["mean"][index]),
                    float(forecast["lower"][index]),
                    float(forecast["upper"][index]),
                ]
            )

        self.client.insert(self.table, rows, column_names=COLUMNS)
