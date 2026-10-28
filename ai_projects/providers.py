"""Small HTTP adapters. Keys are read only at request time and never logged."""
import json
import os
import ssl
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ProviderError(RuntimeError):
    pass


class APIModel:
    def __init__(self, provider: str, model: str | None = None):
        if provider not in {"openai", "anthropic"}:
            raise ValueError("Provider must be openai or anthropic")
        self.provider = provider
        self.model = model or os.getenv(
            "OPENAI_MODEL" if provider == "openai" else "ANTHROPIC_MODEL",
            "gpt-4.1-mini" if provider == "openai" else "claude-sonnet-4-5",
        )

    def complete(self, system: str, messages: list[dict]) -> str:
        key_name = "OPENAI_API_KEY" if self.provider == "openai" else "ANTHROPIC_API_KEY"
        key = os.getenv(key_name)
        if not key:
            raise ProviderError(f"Set {key_name} securely to use {self.provider}")
        if self.provider == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {key}"}
            payload = {"model": self.model, "messages": [{"role": "system", "content": system}, *messages]}
        else:
            url = "https://api.anthropic.com/v1/messages"
            headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
            payload = {"model": self.model, "max_tokens": 2048, "system": system, "messages": messages}
        request = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", **headers})
        try:
            context = ssl.create_default_context(cafile=os.getenv("SSL_CERT_FILE"))
            with urlopen(request, timeout=60, context=context) as response:
                data = json.load(response)
        except HTTPError as exc:
            # Response bodies may contain sensitive request details; do not echo them.
            raise ProviderError(f"{self.provider} returned HTTP {exc.code}") from None
        except (URLError, TimeoutError, OSError, ValueError):
            raise ProviderError(f"{self.provider} request failed; check connectivity and TLS configuration") from None
        try:
            if self.provider == "openai":
                result = data["choices"][0]["message"]["content"]
            else:
                result = "\n".join(block["text"] for block in data["content"] if block["type"] == "text")
            if not isinstance(result, str) or not result.strip():
                raise ValueError("empty response")
            return result
        except (KeyError, IndexError, TypeError, ValueError):
            raise ProviderError(f"{self.provider} returned an unexpected response format") from None
