import logging
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yaml

from src.clickhouse_sink import ClickHouseSink
from src.exporter import ForecastExporter
from src.forecaster import MoiraiForecaster
from src.preprocess import preprocess_metric
from src.prometheus_api import PrometheusClient

logger = logging.getLogger("vllm-forecaster")


def load_config():
    path = os.getenv("CONFIG_PATH", "config.yaml")
    with open(path, "r") as file:
        return yaml.safe_load(file)


def apply_clip(prediction, clip):
    """clip: [low, high]; either side may be null."""
    if not clip:
        return prediction

    low, high = (list(clip) + [None, None])[:2]
    if low is None and high is None:
        return prediction

    return {
        key: np.clip(values, low, high) for key, values in prediction.items()
    }


def update_buffer(buffers, name, prometheus, query, step, end_time, history, overlap):
    """
    Rolling raw-sample buffer per metric.

    First call backfills the whole history (chunked). Later calls fetch only
    the newest samples (plus a small overlap) and trim to the history window.
    """
    cutoff = end_time - history
    old = buffers.get(name)

    if old is None or old.empty:
        start = cutoff
    else:
        last = old["timestamp"].iloc[-1].to_pydatetime()
        start = max(cutoff, last - overlap)

    new = prometheus.query_range(
        query=query,
        start=start.timestamp(),
        end=end_time.timestamp(),
        step=step,
    )

    frames = [f for f in (old, new) if f is not None and not f.empty]

    if not frames:
        return new

    merged = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset="timestamp", keep="last")
        .sort_values("timestamp")
    )
    merged = merged[merged["timestamp"] >= pd.Timestamp(cutoff)]
    merged = merged.reset_index(drop=True)

    buffers[name] = merged
    return merged


def forecast_metric_name(name):
    """vllm:num_requests_running -> vllm:forecast_num_requests_running"""
    prefix, sep, rest = name.partition(":")
    return f"{prefix}:forecast_{rest}" if sep else f"forecast_{name}"


def build_sink(clickhouse_config):
    if not clickhouse_config or not clickhouse_config.get("enabled", False):
        logger.info("ClickHouse sink disabled")
        return None

    return ClickHouseSink(
        host=os.getenv("CLICKHOUSE_HOST", clickhouse_config.get("host", "localhost")),
        port=int(os.getenv("CLICKHOUSE_PORT", clickhouse_config.get("port", 8123))),
        username=os.getenv("CLICKHOUSE_USER", clickhouse_config.get("username", "default")),
        password=os.getenv("CLICKHOUSE_PASSWORD", clickhouse_config.get("password", "")),
        database=clickhouse_config.get("database", "monitoring"),
        table=clickhouse_config.get("table", "vllm_forecasts"),
        secure=clickhouse_config.get("secure", False),
        ttl_days=clickhouse_config.get("ttl_days", 90),
    )


def main():

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config()

    prometheus = PrometheusClient(
        os.getenv("PROMETHEUS_URL", config["prometheus"]["url"])
    )

    model_config = config["model"]

    forecaster = MoiraiForecaster(
        checkpoint=model_config["checkpoint"],
        prediction_length=model_config["prediction_length"],
        context_length=model_config["context_length"],
        num_samples=model_config["num_samples"],
        batch_size=model_config["batch_size"],
        patch_size=model_config.get("patch_size", "auto"),
    )

    exporter = None
    exporter_config = config.get("exporter", {})
    if exporter_config.get("enabled", True):
        exporter = ForecastExporter(
            host=exporter_config["host"],
            port=exporter_config["port"],
        )
        exporter.start()

    sink = build_sink(config.get("clickhouse"))

    forecast_config = config["forecast"]
    refresh_interval = forecast_config["refresh_interval_seconds"]
    minimum_points = forecast_config["minimum_data_points"]

    step = config["prometheus"].get("sampling_interval", "15s")
    history = timedelta(hours=config["prometheus"]["history_hours"])
    overlap = timedelta(minutes=forecast_config.get("incremental_overlap_minutes", 10))

    for metric in config["metrics"]:
        if metric.get("type") == "counter_rate" and "rate(" not in metric["query"]:
            logger.warning(
                "%s is marked counter_rate but its query has no rate()/irate(). "
                "Never forecast a raw counter.",
                metric["name"],
            )

    buffers = {}

    try:
        while True:

            cycle_started = time.monotonic()

            # Floor to the minute so every 1-minute bucket is complete.
            end_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)

            logger.info("Starting forecasting cycle (end=%s)", end_time.isoformat())

            for metric in config["metrics"]:

                name = metric["name"]
                forecast_name = metric.get("forecast_name") or forecast_metric_name(name)

                try:
                    df = update_buffer(
                        buffers,
                        name,
                        prometheus,
                        metric["query"],
                        step,
                        end_time,
                        history,
                        overlap,
                    )

                    if df.empty:
                        logger.warning("No data available for %s", name)
                        continue

                    series = preprocess_metric(
                        df,
                        minimum_points=minimum_points,
                        context_length=model_config["context_length"],
                        max_gap_minutes=forecast_config.get("max_gap_minutes", 5),
                        max_fill_ratio=forecast_config.get("max_fill_ratio", 0.2),
                        max_staleness_minutes=forecast_config.get(
                            "max_staleness_minutes", 5
                        ),
                        end_time=end_time,
                    )

                    prediction = forecaster.predict(series)
                    prediction = apply_clip(prediction, metric.get("clip"))

                    # Forecast step i refers to the 1-minute bucket that starts
                    # (i + 1) minutes after the last observed bucket.
                    target_timestamps = pd.date_range(
                        start=series.index[-1] + pd.Timedelta(minutes=1),
                        periods=len(prediction["mean"]),
                        freq="1min",
                    )

                    if exporter is not None:
                        exporter.update_forecast(forecast_name, prediction, target_timestamps)

                    if sink is not None:
                        try:
                            sink.write_forecast(name, forecast_name, target_timestamps, prediction)
                        except Exception:
                            logger.exception("ClickHouse insert failed for %s", name)

                    logger.info(
                        "Forecast generated for %s (%s -> %s)",
                        name,
                        target_timestamps[0].isoformat(),
                        target_timestamps[-1].isoformat(),
                    )

                except Exception:
                    # One failing metric must not stop the others.
                    logger.exception("Forecast failed for %s", name)

            elapsed = time.monotonic() - cycle_started
            wait = max(1.0, refresh_interval - elapsed)

            logger.info("Cycle took %.1fs, waiting %.1fs", elapsed, wait)
            time.sleep(wait)

    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        if exporter is not None:
            exporter.stop()


if __name__ == "__main__":
    main()
