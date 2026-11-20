# Qwen vLLM — two independent servers + regular router + observability

This restores the **previous architecture**: two independent Qwen vLLM OpenAI-compatible servers (one on each machine), a regular vLLM Router in front of them, and a monitoring stack that scrapes each backend and the router independently.

There is **no prefill/decode disaggregation, NIXL, KV transfer, or P/D mode** in this project.

## 1. Architecture

```text
                 Client / Locust / curl
                          |
                          v
                 Router :8001 (HTTP)
                 Router metrics :9000
                    /          \
                   /            \
          Server A :8000     Server B :8000
          Machine A          Machine B
             |                   |
             +---------+---------+
                       |
          Prometheus :9090 scrapes:
          - Server A /metrics
          - Server B /metrics
          - Router /metrics
          - DCGM exporter (optional GPU metrics)
                       |
                 Grafana :3000

          vLLM traces from both servers
                       |
                 Jaeger :16686
                 OTLP :4317/:4318
```

Each vLLM server is independent and serves the same model. The router distributes requests using the configured policy (`round_robin` by default; `consistent_hash` is also available).

## 2. Folder structure

```text
qwen-vllm-previous-architecture/
├── .env.example
├── README.md
├── servers/
│   ├── docker-compose.yml
│   ├── .env.server-a.example
│   └── .env.server-b.example
├── router/
│   └── docker-compose.yml
├── monitoring/
│   ├── docker-compose.yml
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/prometheus.yml
│       │   └── dashboards/dashboards.yml
│       └── dashboards/qwen-vllm-overview.json
├── load-testing/
│   ├── requirements.txt
│   └── locustfile.py
└── scripts/
    ├── start-server.ps1
    ├── start-router.ps1
    ├── start-monitoring.ps1
    ├── test-router.ps1
    └── stop-all.ps1
```

## 3. Prerequisites

- Docker Desktop / Docker Engine with NVIDIA Container Toolkit configured.
- NVIDIA GPU and a working `nvidia-smi` on each inference machine.
- Both machines reachable over the same LAN/VPN, or otherwise routable to each other.
- TCP port `8000` reachable from the router machine on both inference machines.
- TCP ports `9090`, `3000`, `16686`, `4317`, `4318` as needed on the monitoring machine.
- Model access to `Qwen/Qwen2.5-0.5B-Instruct`. For a gated/private model, configure a Hugging Face token.

## 4. Configure the two server machines

Copy the project to **both** inference machines.

### Machine A
From `servers/`:

```powershell
Copy-Item .env.server-a.example .env
```

Edit `servers/.env`:
- `SERVER_NAME=server-a`
- `HOST_PORT=8000`
- `GPU_MEMORY_UTILIZATION=0.35` (adjust for your GPU)
- `MONITORING_HOST` = the LAN IP of the machine running Jaeger, e.g. `192.168.x.x`
- `HF_TOKEN` if required.

Start:

```powershell
docker compose up -d
docker compose logs -f vllm-server
```

### Machine B
Use the same `servers/docker-compose.yml`, but copy:

```powershell
Copy-Item .env.server-b.example .env
```

Set its own `SERVER_NAME`, `HOST_PORT`, `MONITORING_HOST`, GPU memory utilization, and token. Start it with the same `docker compose up -d`.

Both servers expose `http://<machine-ip>:8000/v1` and `http://<machine-ip>:8000/metrics`.

> The two servers can use different GPUs and GPU-memory settings. Keep the served model name the same on both (`qwen-0.5b`) so the router/client use one model name.

## 5. Configure router and monitoring machine

On the machine that will run the router and monitoring stack, copy `.env.example` to `.env` in the project root.

Set:
- `LOCAL_VLLM_URL=http://<SERVER_A_LAN_IP>:8000`
- `REMOTE_VLLM_URL=http://<SERVER_B_LAN_IP>:8000`
- `ROUTER_POLICY=round_robin` (or `consistent_hash`)
- `MONITORING_HOST_IP=<LAN_IP_OF_THIS_MACHINE>`

For example, do not retain the example IPs: replace them with the current addresses from `ipconfig` / `ip addr`.

### Generate Prometheus targets and start monitoring

Run this from the project root, replacing the example addresses:

```powershell
.\\scripts\\configure-prometheus.ps1 -ServerA 192.168.1.101:8000 -ServerB 192.168.1.102:8000
cd monitoring
docker compose --env-file ../.env up -d
cd ..
```

### Start router

```powershell
cd router
docker compose --env-file ../.env up -d
docker compose --env-file ../.env logs -f vllm-router
```

Router endpoint: `http://<ROUTER_MACHINE_IP>:8001/v1`

Router metrics: `http://<ROUTER_MACHINE_IP>:9000/metrics`

## 6. Verify each layer

From the router machine:

```powershell
curl.exe http://<SERVER_A_LAN_IP>:8000/health
curl.exe http://<SERVER_B_LAN_IP>:8000/health
curl.exe http://localhost:8001/health
curl.exe http://localhost:9000/metrics
curl.exe http://localhost:9090/-/ready
```

Test a chat completion:

```powershell
$body = @{
  model = "qwen-0.5b"
  messages = @(@{ role = "user"; content = "Explain what a GPU KV cache is in two sentences." })
  max_tokens = 100
  temperature = 0.2
} | ConvertTo-Json -Depth 8

Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8001/v1/chat/completions" `
  -ContentType "application/json" `
  -Body $body
```

The router returning HTTP 200 confirms a completion was served; it does not by itself prove that both workers received traffic. Check router metrics and each server's own metrics/logs.

## 7. Grafana

Open `http://<MONITORING_HOST_IP>:3000`.

Default credentials in this bundle:
- user: `admin`
- password: value of `GRAFANA_ADMIN_PASSWORD` in root `.env`

The dashboard `Qwen vLLM — Two Servers + Router` is provisioned automatically. Prometheus is preconfigured as the default data source.

Dashboard panels include:
- request rate by server
- running and waiting requests by server
- prefill time average
- TTFT p95
- end-to-end latency p95
- generated tokens/sec
- GPU KV cache usage
- router request rate, active workers, retries and processed requests by worker
- GPU utilization and memory (when DCGM exporter is enabled and working)

Metric names can differ across vLLM versions/builds. If a panel is empty, first inspect `/metrics` on that server and use Grafana Explore to confirm the exact metric name. Router metrics are separate from backend vLLM metrics.

## 8. Jaeger / tracing

Jaeger UI: `http://<MONITORING_HOST_IP>:16686`

Both server `.env` files point vLLM's OTLP trace exporter at the monitoring host's `4318` endpoint. Make sure TCP 4318 is reachable from both inference machines. Detailed traces can add overhead; turn off detailed collection when not profiling.

## 9. Load testing with Locust

On the client machine:

```powershell
cd load-testing
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:TARGET_URL = "http://<ROUTER_MACHINE_IP>:8001"
$env:MODEL_NAME = "qwen-0.5b"
locust -f locustfile.py
```

Open Locust at `http://localhost:8089`. Start with a small user count and ramp gradually. The Locust client sends requests only to the router, not directly to either worker.

## 10. Firewall and network checklist

- Router machine -> Server A: TCP 8000
- Router machine -> Server B: TCP 8000
- Prometheus machine -> both servers: TCP 8000
- Prometheus machine -> router: TCP 9000
- Both servers -> monitoring machine: TCP 4318 (OTLP HTTP)
- Client -> router: TCP 8001
- Browser -> Grafana: TCP 3000
- Browser -> Jaeger: TCP 16686

Do not expose these ports to the public internet. Use a trusted LAN/VPN and firewall rules.

## 11. Important notes

- This is the previous **regular multi-worker router architecture**. It is not the later P/D disaggregation experiment.
- `consistent_hash` is a routing policy, not proof of session affinity unless the router version's supported request-key behavior is configured and verified. With only one worker in a pool, you cannot observe distribution.
- The Prometheus configuration scrapes Server A, Server B and router as separate jobs so backend metrics remain distinguishable.
- GPU utilization/memory panels depend on the NVIDIA DCGM exporter. If it cannot access the GPU, vLLM metrics can still work while GPU panels remain empty.
- This bundle is a clean, internally consistent project scaffold. It has not been run against your relocated machines; update IPs, test reachability, and validate the metric names exposed by your installed image.
