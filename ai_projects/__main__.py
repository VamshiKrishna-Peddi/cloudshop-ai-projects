import argparse
import json
import shutil
import sys
from pathlib import Path

from .evaluate import ROOT, TASK, evaluate, regressions
from .providers import APIModel, ProviderError
from .ollama import OllamaModel, OllamaEmbeddings
from .rag.semantic import SemanticIndex
from .rag.pipeline import answer
from .rag.retrieval import Index
from .task_agent.agent import Agent, OfflineInventoryModel
from .task_agent.tools import Tools


def main(argv=None):
    parser = argparse.ArgumentParser(description="Task automation agent and RAG prototype")
    commands = parser.add_subparsers(dest="command", required=True)
    agent = commands.add_parser("agent", help="Run a task in a bounded workspace")
    agent.add_argument("--workspace", type=Path, default=Path(".local/agent"))
    agent.add_argument("--task", default=TASK)
    agent.add_argument("--demo", action="store_true", help="Copy sample inventory if not present")
    agent.add_argument("--max-steps", type=int, default=10)
    agent.add_argument("--allow-host", action="append", default=[])
    rag = commands.add_parser("rag", help="Index documents and answer a question")
    rag.add_argument("question")
    rag.add_argument("--corpus", type=Path, default=ROOT / "examples/knowledge")
    rag.add_argument("--index", type=Path, help="Save a rebuilt index here, or load with --load-index")
    rag.add_argument("--load-index", action="store_true")
    rag.add_argument("--no-retrieval", action="store_true")
    rag.add_argument("--top-k", type=int, default=3)
    evaluation = commands.add_parser("evaluate", help="Run task and RAG accuracy evaluations")
    evaluation.add_argument("--runs", type=int, default=1)
    evaluation.add_argument("--output", type=Path, default=Path(".local/evaluation.json"))
    evaluation.add_argument("--baseline", type=Path)
    evaluation.add_argument("--tolerance", type=float, default=0.0)
    for command in (agent, rag, evaluation):
        command.add_argument("--provider", choices=["offline", "ollama", "openai", "anthropic"], default="offline")
        command.add_argument("--model")
        command.add_argument("--ollama-url", help="Local Ollama URL (default: OLLAMA_HOST or http://localhost:11434)")
    for command in (rag, evaluation):
        command.add_argument("--embedding-model", default="nomic-embed-text")
    args = parser.parse_args(argv)
    try:
        model = (OllamaModel(args.model, args.ollama_url) if args.provider == "ollama"
                 else APIModel(args.provider, args.model) if args.provider != "offline" else None)
        embedder = OllamaEmbeddings(args.embedding_model, args.ollama_url) if args.provider == "ollama" and args.command != "agent" else None
        if args.command == "agent":
            if model is None and args.task != TASK:
                raise ValueError("Offline mode supports only the inventory demo task; select an API provider for custom tasks")
            tools = Tools(args.workspace, args.allow_host)
            if args.demo and not (tools.root / "inventory.csv").exists():
                shutil.copy(ROOT / "examples/agent/inventory.csv", tools.root)
            result = Agent(model or OfflineInventoryModel(), tools, args.max_steps).run(args.task).to_dict()
            print(json.dumps(result, indent=2))
            return 0 if result["completed"] and not any(not event["ok"] for event in result["trace"]) else 1
        if args.command == "rag":
            if args.load_index and not args.index:
                raise ValueError("--load-index requires --index")
            if embedder:
                index = SemanticIndex.load(args.index, embedder) if args.load_index else SemanticIndex.build(args.corpus, embedder)
            else:
                index = Index.load(args.index) if args.load_index else Index.build(args.corpus)
            if args.index and not args.load_index:
                index.save(args.index)
            print(json.dumps(answer(args.question, index, model, not args.no_retrieval, args.top_k), indent=2))
            return 0
        result = evaluate(model, args.runs, embedder)
        failures = regressions(result, json.loads(args.baseline.read_text()), args.tolerance) if args.baseline else []
        result["regressions"] = failures
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"mode": result["mode"], "metrics": result["metrics"], "regressions": failures, "output": str(args.output)}, indent=2))
        # An accuracy evaluation may report partial success; CI must fail on errors.
        return 1 if failures or any(result["metrics"][key] < 1 for key in ("agent_success_rate", "rag_accuracy", "retrieval_recall_at_3", "citation_validity")) else 0
    except (ProviderError, ValueError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
