# CloudShop AI Projects

Two Python projects built around a small fictional shop, with runnable offline demos, local Ollama models plus optional OpenAI and Anthropic integrations, and repeatable evaluations. Python 3.11+ is required. The CLI core uses the standard library. The optional browser workspace uses Streamlit and pypdf from `requirements.txt`. No paid API key or database is required for local Ollama use.

## CloudShop AI Workspace — browser app on macOS

The Streamlit dashboard contains **RAG Assistant** and **AI Agent** pages with sidebar navigation and Ollama/model status. It uses the existing `Agent`, `Tools`, `OllamaModel`, `OllamaEmbeddings`, `SemanticIndex`, and RAG answer pipeline. No paid providers are used by the browser app.

From your local project checkout (for example `~/cloudshop-ai-projects`):

```bash
cd "$HOME/cloudshop-ai-projects"  # adjust if you cloned elsewhere
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama list
```

Make sure `llama3.2` and `nomic-embed-text` are installed. If needed:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Keep Ollama running (`ollama serve` in a separate terminal if it is not already running), then start the UI from the repository root:

```bash
source .venv/bin/activate
python -m streamlit run streamlit_app.py --server.address localhost --server.port 8501
```

Open **http://localhost:8501** in your Mac browser. The app connects only to **http://localhost:11434**, explicitly uses `llama3.2` for generation/tool actions, and `nomic-embed-text` for embeddings. Click **Refresh Ollama status** after starting Ollama or downloading a model. The browser app and Ollama must run on the same machine. Stop Streamlit with Ctrl+C. If port 8501 is occupied, stop the other process before restarting.

**RAG Assistant:** upload TXT, Markdown (`.md`), or text-based PDF documents, then click **Build knowledge library**. Ask questions in the chat and expand the source panel to inspect cited chunk IDs, extracted text, and cosine similarity scores. The existing 100-word/20-word-overlap chunking and dense-vector retrieval are reused. Each question is answered independently; chat history is displayed but is not used to resolve follow-up pronouns. Changing uploaded files clears the prior index and conversation. Rebuilding is explicit and a failed rebuild removes the previous index.

Up to 10 documents, 5 MB each, 100 PDF pages, 500,000 extracted characters per document, and 1,000,000 combined characters are accepted. Password-protected PDFs, scanned PDFs without OCR, empty text, malformed files, and unsupported formats produce actionable errors. PDF citations identify the file and chunk offset, not page numbers. Uploaded paths are normalized and duplicate names receive distinct source IDs.

The minimum retrieval score defaults to 0.30 and is adjustable. If no chunk meets it, the app abstains without asking the model to guess. Uncited answers, invalid citation IDs, and recognized model abstentions are shown as unsupported. Similarity and valid citation IDs are **not proof of factual support**; verify answers against source text. These are useful prototype checks, not a semantic entailment verifier.

**AI Agent:** upload a UTF-8 CSV with `product,stock,reorder_level` headers (use **Download sample CSV**), enter a task, and click **Run inventory agent**. Preview rows and calculated order quantities before execution. Each run invokes the existing bounded agent with Ollama in a fresh temporary workspace, shows its tool results/errors, and provides a Markdown report download. The requested task can guide analysis; this page always also asks for the standard reorder report. Only that report is available for download. Step details appear after execution finishes, not as a live stream.

Reorder arithmetic is shared Python logic: `max(0, reorder_level - stock)`, with products at or above their reorder level excluded from the report. Blank/duplicate product names, missing/duplicate headers, malformed rows, negative/fractional/nonnumeric values, and inventories over 64 KB are rejected. A deterministic `inventory_report` tool writes the report. The UI keeps input read-only, rejects model-written report contents with incorrect quantities, and validates the saved report before displaying/downloading it. An agent completion claim without a verified report is shown as incomplete. If a later model request fails, completed tool steps and any verified report remain available.

Documents, vectors, conversation, and output are session-local; no shared vector database or paid service is used. Temporary extraction/agent directories are cleaned up. RAG indices and results stay in Streamlit session memory until cleared, replaced, or the session ends. Traces can contain uploaded content. Streamlit usage telemetry is disabled and the server binds to localhost by default. This is a local development UI, with no authentication or production deployment configuration.

Run all tests and offline evaluations in the activated virtual environment:

```bash
python -m unittest discover -s tests -v
python -m ai_projects evaluate --runs 3 --baseline evaluations/offline_baseline.json
```

Tests cover the existing CLI/providers plus PDF/text extraction, semantic-index integration, abstention/citations, CSV validation, calculation boundaries, agent tool recovery and report integrity. Streamlit AppTest exercises page navigation, uploads, chat/sources, document-change invalidation, and report downloads. Model calls are mocked in tests; no paid API or downloaded LLM is needed. Install `requirements.txt` before running the complete test suite. CI installs these dependencies as well.

The browser changes are local until explicitly committed and pushed with approval; `git pull` on your Mac will not retrieve unpushed changes.

## Transfer the complete project to your Mac with GitHub

The project is prepared locally in this workspace. Creating a private repository and pushing require your approval; preparing a commit does not upload files. Cloning the existing remote does not include these changes until they are pushed.

After the approved private repository has been created and the project pushed, install GitHub CLI on your Mac and authenticate using the browser flow (never paste tokens into chat):

```bash
brew install git gh python
gh auth login --hostname github.com --git-protocol https --web
```

Then clone the private repository. Replace `YOUR_USERNAME/YOUR_PRIVATE_REPOSITORY` with the exact repository name provided after the approved push. Use a new destination if this directory already contains files:

```bash
mkdir -p "$HOME/cloudshop-local"
gh repo clone YOUR_USERNAME/YOUR_PRIVATE_REPOSITORY "$HOME/cloudshop-local/cloudshop"
cd "$HOME/cloudshop-local/cloudshop"
python3 --version  # Python 3.11+
python3 -m pip install -r requirements.txt  # preferably in the virtual environment above
python3 -m unittest discover -s tests -v
```

Keep using this directory for the application commands below. The repository includes both Python projects, examples, tests, documentation, and the existing JavaScript workspace. The CLI core uses the standard library; install `requirements.txt` for the browser app and complete test suite. API keys, local environment files, dependencies, caches, and generated reports must remain untracked. Ollama models are installed separately on your Mac, not stored in Git.

## Free local models on macOS (Ollama)

Use a Mac supported by the current Ollama release (macOS 14+; Apple Silicon is recommended). Allow several GB of disk space for model downloads and enough available RAM to load the model; 8 GB is a practical starting point for the default 3B model, with more RAM improving headroom. No API subscription or key is required. Internet access is needed for installation and the initial model downloads; subsequent local inference can run without Internet access.

If Ollama is already running with both models installed, skip installation and downloads and start with step 4. Your Mac’s `localhost` is separate from this cloud workspace; run the applications on your Mac to reach its Ollama server.

1. Install Homebrew using the instructions at https://brew.sh if it is not already installed. Then install Python and Ollama:

   ```bash
   brew install python ollama
   python3 --version  # must be 3.11 or newer
   ```

2. Start the local server in a terminal and leave it running:

   ```bash
   ollama serve
   ```

   Alternatively, use `brew services start ollama` to run it in the background. Choose one startup method. If the official Ollama app is already running, it provides the server; do not start another copy. Default address: `http://localhost:11434`.

3. In another terminal, download and verify both models:

   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ollama list
   curl http://localhost:11434/api/tags
   ```

4. Clone the approved private repository as described above. For the full test suite, activate the virtual environment and install `requirements.txt` as described in the browser setup:

   ```bash
   cd "$HOME/cloudshop-local/cloudshop"
   python3 -m unittest discover -s tests -v
   ```

5. Run the task agent with local llama3.2:

   ```bash
   python3 -m ai_projects agent --demo --provider ollama --model llama3.2 \
     --ollama-url http://localhost:11434
   cat .local/agent/report.md
   ```

   This keeps the same list/read/write/HTTPS tool loop. JSON mode constrains local model actions to the existing tool protocol. For a custom task:

   ```bash
   python3 -m ai_projects agent --provider ollama --model llama3.2 \
     --workspace .local/agent --task "Read inventory.csv and write a summary to summary.md."
   ```

6. Run RAG with local generation and semantic embeddings:

   ```bash
   python3 -m ai_projects rag "What is the return window?" \
     --provider ollama --model llama3.2 --embedding-model nomic-embed-text \
     --ollama-url http://localhost:11434 \
     --corpus examples/knowledge --index .local/ollama-index.json
   ```

   Reuse the saved document vectors on later questions (query embedding and generation still require the server):

   ```bash
   python3 -m ai_projects rag "How long does express shipping take?" \
     --provider ollama --model llama3.2 --embedding-model nomic-embed-text \
     --ollama-url http://localhost:11434 \
     --load-index --index .local/ollama-index.json
   ```

7. Run repeated local-model evaluations and compare a future run:

   ```bash
   python3 -m ai_projects evaluate --provider ollama --model llama3.2 \
     --embedding-model nomic-embed-text --runs 3 --output .local/ollama-baseline.json
   python3 -m ai_projects evaluate --provider ollama --model llama3.2 \
     --embedding-model nomic-embed-text --runs 3 \
     --baseline .local/ollama-baseline.json --output .local/ollama-current.json
   ```

   Evaluations use the local model for the agent and for RAG with/without retrieval. They may take several minutes. A quality failure returns exit status 1 and still saves the detailed report; this is not necessarily a setup failure. Small models can produce incorrect actions or answers. Compare local results only with a baseline using the same generation and embedding models. The checked-in offline baseline is for deterministic tests, not for Ollama.

Local generation defaults to `llama3.2`; embeddings default to `nomic-embed-text` whenever `--provider ollama` is selected. The default provider remains `offline` to keep tests and demos independent of a server. There is no paid-provider fallback. Set `--ollama-url http://localhost:11434` (or `OLLAMA_HOST` with a full local HTTP URL) if needed; only loopback hosts are supported. `OLLAMA_MODEL` can override the local generation default. No API key is read by the Ollama adapters, and local traffic bypasses inherited HTTP proxies.

Troubleshooting: connection errors mean the server is unavailable; start it and verify `/api/tags`. HTTP 404 generally means a missing model; repeat the matching `ollama pull`. The first request may be slow while loading a model. If requests time out, close memory-heavy applications and inspect `ollama ps`. Stop a loaded model with `ollama stop llama3.2`; stop a foreground server with Ctrl+C or a background service with `brew services stop ollama`. If an index type/model mismatch is reported, rebuild without `--load-index`. Rebuild after changing documents, embedding model weights, or embedding dimensions. Keep separate files for TF-IDF (`.local/index.json`) and Ollama (`.local/ollama-index.json`) indices.

The tests mock Ollama's HTTP responses, so they do not download models or prove inference quality on your Mac. Live local inference has not been validated in this cloud workspace.

## Quick start (deterministic offline demos)

Run from the repository root:

```bash
python3 -m ai_projects agent --demo
python3 -m ai_projects rag "How long does express shipping take?" --index .local/index.json
python3 -m ai_projects evaluate --runs 3 --output .local/baseline.json
python3 -m unittest discover -s tests -v
```

The agent reads a CSV, identifies low-stock products, and writes `.local/agent/report.md`. The RAG command retrieves a shipping policy and answers with a chunk citation. Evaluation results include per-case outputs and aggregate metrics in JSON. Generated files stay in the ignored `.local/` directory.

## Project 1: AI agent for task automation

Implementation: [`ai_projects/task_agent`](ai_projects/task_agent). [Project guide](ai_projects/task_agent/README.md).

The agent repeatedly asks a model for an action, executes a tool, supplies the observation, and lets the model choose the next action. It supports listing files, reading and writing workspace files, and fetching JSON from explicitly allowed HTTPS hosts. The loop has a step limit, records tool outcomes, and can recover from malformed actions. The sample task produces a reorder report from inventory data.

For free local custom tasks use Ollama as above. The existing optional paid adapters are retained unchanged; use them only if you explicitly choose to:

```bash
# Supply OPENAI_API_KEY securely in your execution environment first.
python3 -m ai_projects agent --demo --provider openai
# Supply ANTHROPIC_API_KEY securely for Claude.
python3 -m ai_projects agent --demo --provider anthropic
python3 -m ai_projects agent --provider openai --workspace .local/agent \
  --task "Read inventory.csv and write a summary to summary.md."
```

Model IDs can be selected with `--model`, `OPENAI_MODEL`, or `ANTHROPIC_MODEL`. Defaults are `gpt-4.1-mini` and `claude-sonnet-4-5`; availability depends on your API account. Provider adapters call the HTTPS APIs directly rather than depending on SDKs. Network settings must permit `api.openai.com` or `api.anthropic.com`. API calls can incur charges.

Offline mode uses a fixed inventory policy to demonstrate the same tool loop. It is not an LLM and does not support arbitrary tasks. A final model response is a completion claim, not proof that the task succeeded; evaluations inspect the resulting artifact independently.

## Project 2: retrieval augmented generation

Implementation: [`ai_projects/rag`](ai_projects/rag). [Project guide](ai_projects/rag/README.md).

Markdown/text documents are chunked with overlap and ranked by cosine similarity. Ollama mode uses normalized `nomic-embed-text` dense embeddings with separate search-document/query prefixes; offline and existing paid-provider modes retain TF-IDF vectors. Retrieved chunks are supplied to the selected model, which is instructed to answer only from context and cite chunk IDs. Offline mode selects a relevant sentence instead of generating text with an LLM. Offline mode uses lexical vector search; Ollama mode uses neural embeddings.

```bash
python3 -m ai_projects rag "What is the return window?" --provider anthropic
python3 -m ai_projects rag "What is the return window?" --corpus examples/knowledge --index .local/index.json
python3 -m ai_projects rag "What is the return window?" --load-index --index .local/index.json
python3 -m ai_projects rag "What is the return window?" --no-retrieval
```

## Evaluations and regressions

```bash
python3 -m ai_projects evaluate --runs 3 --output .local/baseline.json
python3 -m ai_projects evaluate --runs 3 --baseline .local/baseline.json --output .local/current.json
# Optional paid end-to-end model experiment:
python3 -m ai_projects evaluate --provider openai --runs 3 --output .local/openai.json
```

CI compares each run against the checked-in `evaluations/offline_baseline.json`, generated from the verified offline fixture run. Update that baseline deliberately when changing expected behavior.

The agent evaluation checks the saved report for required products, quantities, total, and absence of the in-stock product. RAG evaluation uses five expected-answer fixtures, measuring answer accuracy, source retrieval recall@3, and citation validity. Each question runs with and without retrieval to measure the observed accuracy difference. Regression comparison checks quality metrics against a saved result from the same mode/provider/model; `--tolerance` permits a specified absolute decrease. Exit status is 1 on a failed quality check or regression and 2 on a configuration/request error.

Offline fixture evaluations pass at 100% on this tiny dataset; the no-context offline baseline always abstains. This validates wiring and deterministic fixtures, **not real LLM accuracy or a generalizable improvement**. Live model accuracy, latency, cost, prompt injection resistance, and broader task coverage require separate measurements. Citation validation checks IDs, not whether a citation truly supports an answer. Expected-substring scoring is intentionally simple and can miss subtle errors.

## Layout

- `ai_projects/task_agent/`: tool loop and tool implementations.
- `ai_projects/rag/`: index persistence, retrieval, and answer generation.
- `ai_projects/providers.py`: OpenAI and Anthropic HTTP adapters.
- `ai_projects/evaluate.py`, `evaluations/`: metrics and fixtures.
- `examples/`: sample CSV and fictional knowledge documents.
- `tests/`: offline unit and integration tests; API adapters are tested with mocked responses.

The existing `packages/` JavaScript UI/configuration workspace is preserved. Its checks use Node >=24 and npm 11.17.0 (`npm run lint`, `npm run check-types`). There are no storefront apps in this checkout.
