from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread
import time


def _escape(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


class ForecastExporter:
    """Optional Prometheus-format endpoint for the latest forecasts."""

    def __init__(self, host="0.0.0.0", port=9108):

        self.host = host
        self.port = port

        self.forecasts = {}
        self._lock = Lock()
        self.thread = None

        exporter = self

        class MetricsHandler(BaseHTTPRequestHandler):

            def do_GET(self):

                if self.path != "/metrics":
                    self.send_response(404)
                    self.end_headers()
                    return

                body = exporter.render().encode("utf-8")

                self.send_response(200)
                self.send_header(
                    "Content-Type", "text/plain; version=0.0.4"
                )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                return

        self.server = ThreadingHTTPServer((self.host, self.port), MetricsHandler)

    def update_forecast(self, forecast_name, forecast, target_timestamps):
        """
        target_timestamps: iterable of tz-aware pandas Timestamps / datetimes,
        one per forecast step.
        """

        entry = {
            "mean": [float(x) for x in forecast["mean"]],
            "lower": [float(x) for x in forecast["lower"]],
            "upper": [float(x) for x in forecast["upper"]],
            "targets": [t.timestamp() for t in target_timestamps],
        }

        with self._lock:
            self.forecasts[forecast_name] = entry

    def render(self):

        with self._lock:
            snapshot = dict(self.forecasts)

        lines = []

        # metric_name here is the forecast name, e.g.
        # vllm:forecast_num_requests_running. `horizon` = minutes ahead.
        for forecast_name, entry in snapshot.items():
            series = (
                (forecast_name, entry["mean"], "Forecast mean"),
                (f"{forecast_name}_lower", entry["lower"], "Forecast 10th percentile"),
                (f"{forecast_name}_upper", entry["upper"], "Forecast 90th percentile"),
                (
                    f"{forecast_name}_target_timestamp_seconds",
                    entry["targets"],
                    "Unix time the forecast step refers to",
                ),
            )

            for name, values, help_text in series:
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} gauge")
                for index, value in enumerate(values):
                    lines.append(f'{name}{{horizon="{index + 1}"}} {value}')

        return "\n".join(lines) + "\n"

    def start(self):

        self.thread = Thread(
            target=self.server.serve_forever,
            daemon=True,
            name="forecast-exporter",
        )
        self.thread.start()

        time.sleep(0.5)

        if not self.thread.is_alive():
            raise RuntimeError(
                "Forecast exporter HTTP server failed to start"
            )

        print(
            f"Forecast exporter running at "
            f"http://127.0.0.1:{self.port}/metrics"
        )

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
