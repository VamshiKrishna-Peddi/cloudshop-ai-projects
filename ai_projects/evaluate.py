"""Repeatable artifact and answer evaluations with baseline regression checks."""
import json
import re
import shutil
import tempfile
from pathlib import Path

from .rag.pipeline import answer
from .rag.retrieval import Index
from .rag.semantic import SemanticIndex
from .task_agent.agent import Agent, OfflineInventoryModel
from .task_agent.tools import Tools

ROOT = Path(__file__).resolve().parent.parent
TASK = "Read inventory.csv and write report.md listing only products below reorder_level, their stock, order quantity (reorder_level minus stock), and the total count as 'Products requiring reorder: N'."


def evaluate(model=None, runs=1, embedder=None):
    if runs < 1:
        raise ValueError("runs must be positive")
    cases = json.loads((ROOT / "evaluations/rag_cases.json").read_text())
    index = SemanticIndex.build(ROOT / "examples/knowledge", embedder) if embedder else Index.build(ROOT / "examples/knowledge")
    agent_rows, rag_rows = [], []
    for run in range(runs):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            shutil.copy(ROOT / "examples/agent/inventory.csv", workspace)
            result = Agent(model or OfflineInventoryModel(), Tools(workspace)).run(TASK)
            report = (workspace / "report.md").read_text() if (workspace / "report.md").is_file() else ""
            required = [r"Keyboard[^\n]*stock\s*3[^\n]*order\s*7\b", r"Monitor[^\n]*stock\s*1[^\n]*order\s*3\b", r"Products requiring reorder:\s*2\b"]
            passed = result.completed and all(re.search(pattern, report, re.IGNORECASE) for pattern in required) and "mouse" not in report.lower() and not any(not step["ok"] for step in result.trace)
            agent_rows.append({"run": run + 1, "passed": passed, "completed": result.completed, "steps": len(result.trace), "tool_errors": sum(not s["ok"] for s in result.trace), "report": report})
        for case in cases:
            augmented = answer(case["question"], index, model)
            baseline = answer(case["question"], index, model, retrieve=False)
            expected = case["expected"].lower()
            rag_rows.append({"run": run + 1, "question": case["question"], "expected": case["expected"], "answer": augmented["answer"], "baseline_answer": baseline["answer"], "correct": expected in augmented["answer"].lower(), "baseline_correct": expected in baseline["answer"].lower(), "retrieval_hit": any(hit["source"] == case["source"] for hit in augmented["sources"]), "citation_valid": bool(augmented["citations"]) and not augmented["invalid_citations"]})
    metrics = {"agent_success_rate": sum(row["passed"] for row in agent_rows) / len(agent_rows)}
    for metric, key in [("rag_accuracy", "correct"), ("baseline_accuracy", "baseline_correct"), ("retrieval_recall_at_3", "retrieval_hit"), ("citation_validity", "citation_valid")]:
        metrics[metric] = sum(row[key] for row in rag_rows) / len(rag_rows)
    metrics["retrieval_accuracy_gain"] = metrics["rag_accuracy"] - metrics["baseline_accuracy"]
    return {"mode": "local-ollama" if getattr(model, "provider", None) == "ollama" else "live-api" if model else "offline-deterministic", "embedding_model": getattr(embedder, "model", None), "provider": getattr(model, "provider", None), "model": getattr(model, "model", None), "runs": runs, "metrics": metrics, "agent_cases": agent_rows, "rag_cases": rag_rows}


def regressions(current, baseline, tolerance=0.0):
    if not 0 <= tolerance <= 1:
        raise ValueError("tolerance must be between 0 and 1")
    if (current["mode"], current.get("provider"), current.get("model")) != (baseline["mode"], baseline.get("provider"), baseline.get("model")):
        raise ValueError("Compare baselines from the same mode, provider and model")
    if current.get("embedding_model") != baseline.get("embedding_model"):
        raise ValueError("Compare baselines using the same embedding model")
    keys = ("agent_success_rate", "rag_accuracy", "retrieval_recall_at_3", "citation_validity")
    return [key for key in keys if current["metrics"][key] + tolerance < baseline["metrics"][key]]
