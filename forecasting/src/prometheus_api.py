import logging

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Prometheus rejects range queries above 11,000 points per series.
# Stay safely under that limit.
MAX_POINTS_PER_REQUEST = 10_000

_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_step(step):
    """Convert '15s', '1m', '2h' (or a number of seconds) to seconds."""
    step = str(step).strip()
    if step and step[-1] in _UNITS:
        return int(float(step[:-1]) * _UNITS[step[-1]])
    return int(float(step))


def _empty_frame():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime([], utc=True),
            "value": pd.Series([], dtype="float64"),
        }
    )


class PrometheusClient:

    def __init__(self, prometheus_url, timeout=60):
        self.base_url = prometheus_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=(502, 503, 504),
            allowed_methods=("GET",),
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def query_range(self, query, start, end, step="15s"):
        """
        Fetch [start, end] (unix seconds) at `step` resolution.

        Long ranges are split into chunks that respect Prometheus'
        points-per-series limit and concatenated into one DataFrame
        with columns: timestamp (UTC), value.
        """
        step_s = parse_step(step)

        # Align to the step grid so repeated queries hit the same samples.
        start = start - (start % step_s)

        span = MAX_POINTS_PER_REQUEST * step_s
        frames = []
        cursor = start

        while cursor <= end:
            chunk_end = min(cursor + span, end)

            df = self._query_range_once(query, cursor, chunk_end, step)
            if not df.empty:
                frames.append(df)

            if chunk_end >= end:
                break
            cursor = chunk_end + step_s

        if not frames:
            return _empty_frame()

        return (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset="timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    def _query_range_once(self, query, start, end, step):
        url = f"{self.base_url}/api/v1/query_range"

        params = {
            "query": query,
            "start": start,
            "end": end,
            "step": step,
        }

        response = self.session.get(url, params=params, timeout=self.timeout)

        if response.status_code == 400:
            raise RuntimeError(
                f"Prometheus rejected the query: {response.text}"
            )

        response.raise_for_status()

        result = response.json()

        if result.get("status") != "success":
            raise RuntimeError(f"Prometheus query failed: {result}")

        series = result["data"]["result"]

        if not series:
            return _empty_frame()

        if len(series) > 1:
            # Aggregate in PromQL (sum/max/avg) to avoid this.
            logger.warning(
                "Query returned %d series; using the longest one. "
                "Aggregate in PromQL to forecast the total. Query: %s",
                len(series),
                " ".join(query.split()),
            )

        values = max(series, key=lambda s: len(s["values"]))["values"]

        df = pd.DataFrame(values, columns=["timestamp", "value"])

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        return df
