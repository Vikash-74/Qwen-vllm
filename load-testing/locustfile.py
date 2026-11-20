import os
import uuid
from locust import HttpUser, task, between

MODEL_NAME = os.getenv("MODEL_NAME", "qwen-0.5b")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "128"))
PROMPTS = [
    "Explain the difference between CPU and GPU in simple terms.",
    "What is a KV cache in large language model inference?",
    "Explain batching and why it improves throughput in model serving.",
    "Describe how Prometheus and Grafana work together.",
    "What is time to first token and why is it important?",
    "Explain round-robin load balancing with a simple example.",
]

class QwenRouterUser(HttpUser):
    wait_time = between(0.2, 1.0)

    @task
    def chat_completion(self):
        prompt = PROMPTS[uuid.uuid4().int % len(PROMPTS)]
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.2,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        self.client.post(
            "/v1/chat/completions",
            json=payload,
            headers=headers,
            name="/v1/chat/completions",
        )
