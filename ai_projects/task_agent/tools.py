import json
import os
import ssl
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener, HTTPSHandler


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Redirects are disabled for the API tool")


class Tools:
    def __init__(self, workspace: Path, allowed_hosts=()):
        self.root = workspace.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.allowed_hosts = set(allowed_hosts)

    def path(self, name: str) -> Path:
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError("File path must stay inside the agent workspace")
        return path

    def call(self, name: str, args: dict):
        if not isinstance(args, dict):
            raise ValueError("Tool arguments must be an object")
        if name == "list_files":
            return sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file() and p.resolve().is_relative_to(self.root))[:200]
        if name == "read_file":
            path = self.path(args["path"])
            if path.stat().st_size > 64_000:
                raise ValueError("File exceeds 64 KB")
            return path.read_text()
        if name == "write_file":
            content = args["content"]
            if not isinstance(content, str) or len(content.encode()) > 64_000:
                raise ValueError("Content must be text of at most 64 KB")
            path = self.path(args["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return {"written": args["path"], "bytes": len(content.encode())}
        if name == "get_json":
            url = args["url"]
            parsed = urlparse(url)
            if parsed.scheme != "https" or parsed.hostname not in self.allowed_hosts or parsed.username or parsed.password or parsed.port not in (None, 443):
                raise ValueError("HTTPS URL must use an explicitly allowed host on port 443")
            context = ssl.create_default_context(cafile=os.getenv("SSL_CERT_FILE"))
            opener = build_opener(NoRedirect(), HTTPSHandler(context=context))
            with opener.open(Request(url, headers={"Accept": "application/json"}), timeout=15) as response:
                raw = response.read(64_001)
            if len(raw) > 64_000:
                raise ValueError("API response exceeds 64 KB")
            return json.loads(raw)
        raise ValueError(f"Unknown tool: {name}")
