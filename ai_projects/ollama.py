"""Local Ollama generation and embedding adapters; no API keys required."""
import json
import math
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener

from .providers import ProviderError


class OllamaClient:
    def __init__(self, base_url=None):
        self.base_url = (base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        parsed = urlparse(self.base_url)
        if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
                or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment):
            raise ValueError("Ollama URL must be a local HTTP address, e.g. http://localhost:11434")
        # Loopback traffic must not be sent to an inherited cloud/enterprise proxy.
        self.opener = build_opener(ProxyHandler({}))

    def post(self, endpoint, payload):
        request = Request(self.base_url + endpoint, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=180) as response:
                return json.load(response)
        except HTTPError as exc:
            hint = " Run ollama pull llama3.2 and ollama pull nomic-embed-text." if exc.code == 404 else ""
            raise ProviderError(f"Ollama returned HTTP {exc.code}.{hint}") from None
        except (URLError, OSError, ValueError):
            raise ProviderError("Ollama request failed. Start Ollama with 'ollama serve' and check the local URL and available models.") from None


class OllamaModel:
    provider = "ollama"

    def __init__(self, model=None, base_url=None):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
        self.client = OllamaClient(base_url)

    def _complete(self, system, messages, json_mode=False):
        payload = {"model": self.model, "messages": [{"role": "system", "content": system}, *messages], "stream": False, "options": {"temperature": 0}}
        if json_mode:
            payload["format"] = "json"
        data = self.client.post("/api/chat", payload)
        try:
            text = data["message"]["content"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("empty content")
            return text
        except (KeyError, TypeError, ValueError):
            raise ProviderError("Ollama returned an unexpected chat response") from None

    def complete(self, system, messages):
        return self._complete(system, messages)

    def complete_action(self, system, messages):
        """Constrain agent actions to JSON while leaving RAG answers as prose."""
        return self._complete(system, messages, json_mode=True)


class OllamaEmbeddings:
    def __init__(self, model="nomic-embed-text", base_url=None):
        self.model = model
        self.client = OllamaClient(base_url)

    def embed(self, texts):
        data = self.client.post("/api/embed", {"model": self.model, "input": texts, "truncate": False})
        try:
            vectors = data["embeddings"]
            if not isinstance(vectors, list) or len(vectors) != len(texts) or not vectors:
                raise ValueError("embedding count mismatch")
            normalized = []
            dimension = None
            for row in vectors:
                if not isinstance(row, list) or not row or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in row):
                    raise ValueError("invalid embedding")
                dimension = dimension or len(row)
                norm = math.sqrt(sum(x * x for x in row))
                if len(row) != dimension or not math.isfinite(norm) or norm == 0:
                    raise ValueError("invalid dimensions or norm")
                normalized.append([x / norm for x in row])
            return normalized
        except (KeyError, TypeError, ValueError):
            raise ProviderError("Ollama returned invalid embeddings") from None
