# AI agent for task automation

Run `python3 -m ai_projects agent --demo` from the repository root. Inspect `.local/agent/report.md` and the JSON trace printed to stdout. Running again preserves the input CSV and rewrites the generated report.

`Agent` accepts any object exposing `complete(system, messages) -> str`. APIModel implements this interface for the existing paid APIs. OllamaModel implements it for local llama3.2, with an additional complete_action method that requests JSON output from /api/chat. Run `python3 -m ai_projects agent --demo --provider ollama` after installing Ollama and pulling llama3.2 (see the root README for macOS setup). Actions use a simple JSON text protocol rather than provider-specific function calling, allowing one loop to work with both providers. The model sees the task and accumulated observations, chooses one tool per turn, and ends with a final message. Tool errors return to the model; request/authentication failures stop execution. The default limit is ten model turns; `--max-steps` accepts 1–50.

Tools:

| Tool | Arguments | Behavior |
| --- | --- | --- |
| `list_files` | `{}` | Lists up to 200 workspace files |
| `read_file` | `path` | Reads a UTF-8 file, up to 64 KB |
| `write_file` | `path`, `content` | Creates or overwrites a file inside the workspace |
| `inventory_report` | `path`, `output` | Validates CSV and writes a deterministic reorder report |
| `get_json` | `url` | Fetches JSON over HTTPS, at most 64 KB, 15-second timeout |

External JSON APIs require `--allow-host example.com` plus any cloud network policy grant. Redirects, credentials in URLs, alternate ports, and hosts not explicitly allowed are rejected. An example task is `--task "Fetch https://api.example.com/status and save a summary in status.md" --allow-host api.example.com --provider openai` (replace with your actual API). TLS verification stays enabled and inherited proxy settings are respected. The tool has no configurable authentication headers; APIs requiring specialized authentication need an additional explicit tool.

Paths are resolved against a dedicated workspace and traversal/symlink escapes are rejected. Use a disposable directory: writes can overwrite its files, and file/API content and traces may contain sensitive information. The model receives file contents through the selected API. These controls are not an OS sandbox and do not make adversarial tasks safe; avoid privileged hosts or sensitive workspace data. The agent has no shell-execution tool.

The offline inventory policy demonstrates list → read → compute → write → finish. Real model planning can run freely with `--provider ollama`; the existing OpenAI and Anthropic options remain available. `evaluate` checks the report independently of the model's success claim. Tests also cover malformed-action recovery, step limits, blocked paths, and tool size limits.

The browser workspace uses this same agent and an inventory-specific Tools subclass. See the root README for Streamlit setup. CSV validation and report rendering are shared in `inventory.py`; quantities never depend on language model arithmetic.
