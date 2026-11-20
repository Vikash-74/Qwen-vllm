-- Created automatically by ClickHouseSink on startup; kept here for reference.

CREATE DATABASE IF NOT EXISTS monitoring;

CREATE TABLE IF NOT EXISTS monitoring.vllm_forecasts
(
    source_metric   LowCardinality(String),   -- e.g. vllm:num_requests_running
    forecast_metric LowCardinality(String),   -- e.g. vllm:forecast_num_requests_running
    generated_at    DateTime64(3, 'UTC'),     -- when the forecast was made
    target_ts       DateTime('UTC'),          -- the future minute the value refers to
    horizon         UInt16,                   -- 1..prediction_length (minutes ahead)
    forecast_value  Float32,                  -- mean of the samples
    forecast_lower  Float32,                  -- 10th percentile
    forecast_upper  Float32                   -- 90th percentile
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(target_ts)
ORDER BY (forecast_metric, target_ts, generated_at)
TTL target_ts + INTERVAL 90 DAY;
