import numpy as np
import pandas as pd


def preprocess_metric(
    df,
    minimum_points=100,
    context_length=128,
    max_gap_minutes=5,
    max_fill_ratio=0.2,
    max_staleness_minutes=5,
    end_time=None,
):
    """
    Turn raw Prometheus samples into a clean 1-minute series.

    - drops NaN / +-Inf samples
    - resamples to 1 minute (mean)
    - interpolates only short gaps (<= max_gap_minutes)
    - keeps only the latest contiguous stretch (never bridges long outages)
    - rejects stale data and series that are mostly interpolated
    - returns the last `context_length` points as float32

    `end_time` (tz-aware datetime) is the query end. Samples in the
    still-incomplete final minute are discarded.
    """

    if df.empty:
        raise ValueError("No historical data received from Prometheus")

    df = df.copy()

    df["value"] = df["value"].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["timestamp", "value"])

    if df.empty:
        raise ValueError("No finite values in the received data")

    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.set_index("timestamp")

    last_observation = df.index[-1]

    if end_time is not None:
        end_ts = pd.Timestamp(end_time)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")

        age = end_ts - last_observation
        if age > pd.Timedelta(minutes=max_staleness_minutes):
            raise ValueError(
                f"Data is stale: last sample is {age} old"
            )

    series = df["value"].resample("1min").mean()

    if end_time is not None:
        # The bucket starting at end_time holds a single, partial sample.
        series = series[series.index < end_ts.floor("1min")]

    observed = series.notna()

    # Interpolate short gaps only; long gaps stay NaN.
    filled = series.interpolate(
        method="time",
        limit=max_gap_minutes,
        limit_area="inside",
    )

    # Keep only the latest contiguous stretch with no NaN.
    nan_positions = np.flatnonzero(filled.isna().to_numpy())
    if len(nan_positions):
        filled = filled.iloc[nan_positions[-1] + 1:]

    if len(filled) < minimum_points:
        raise ValueError(
            f"Insufficient contiguous data: {len(filled)} points "
            f"(need {minimum_points})"
        )

    tail = filled.iloc[-context_length:]

    fill_ratio = 1.0 - observed.reindex(tail.index).mean()
    if fill_ratio > max_fill_ratio:
        raise ValueError(
            f"{fill_ratio:.0%} of the context window was interpolated "
            f"(limit {max_fill_ratio:.0%})"
        )

    return tail.astype("float32")
