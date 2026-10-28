import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_projects.evaluate import evaluate, regressions
from ai_projects.providers import APIModel, ProviderError
from ai_projects.rag.pipeline import UNKNOWN, answer
from ai_projects.rag.retrieval import Index
from ai_projects.task_agent.agent import Agent, AgentResult
from ai_projects.task_agent.tools import Tools


class ScriptedModel:
    def __init__(self, actions):
        self.actions = iter(actions)

    def complete(self, system, messages):
        return next(self.actions)


class AgentTests(unittest.TestCase):
    def test_recovers_from_bad_action_and_observes_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            model = ScriptedModel(['not json', '{"tool":"write_file","args":{"path":"result.txt","content":"done"}}', '{"final":"done"}'])
            result = Agent(model, Tools(Path(directory))).run("Write result.txt")
            self.assertTrue(result.completed)
            self.assertFalse(result.trace[0]["ok"])
            self.assertTrue(result.trace[1]["ok"])
            self.assertEqual((Path(directory) / "result.txt").read_text(), "done")

    def test_step_limit_stops_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Agent(ScriptedModel(['{}', '{}']), Tools(Path(directory)), 2).run("task")
            self.assertFalse(result.completed)
            self.assertEqual(len(result.trace), 2)

    def test_traversal_and_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            tools = Tools(Path(directory))
            (Path(directory) / "escape").symlink_to(outside, target_is_directory=True)
            for name in ("../secret", str(Path(outside) / "secret"), "escape/secret"):
                with self.assertRaises(ValueError):
                    tools.call("write_file", {"path": name, "content": "bad"})
            self.assertFalse((Path(outside) / "secret").exists())

    def test_api_tool_requires_explicit_https_host(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = Tools(Path(directory), ["example.com"])
            for url in ("http://example.com", "https://other.example", "https://example.com:444", "https://user@example.com"):
                with self.assertRaises(ValueError):
                    tools.call("get_json", {"url": url})

    def test_oversized_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = Tools(Path(directory))
            with self.assertRaises(ValueError):
                tools.call("write_file", {"path": "huge.txt", "content": "a" * 64001})


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.index = Index([{"id": "returns:0", "source": "returns", "text": "Returns accepted within 30 days."}, {"id": "shipping:0", "source": "shipping", "text": "Shipping takes 5 days."}])

    def test_ranking_and_unknown_terms(self):
        self.assertEqual(self.index.search("Returns")[0]["id"], "returns:0")
        self.assertEqual(self.index.search("xyzzy"), [])
        self.assertEqual(answer("xyzzy", self.index)["answer"], UNKNOWN)

    def test_persistence_and_ablation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            self.index.save(path)
            restored = Index.load(path)
            self.assertEqual(self.index.search("shipping"), restored.search("shipping"))
        result = answer("Returns", self.index)
        self.assertIn("30 days", result["answer"])
        self.assertEqual(result["citations"], ["returns:0"])
        self.assertEqual(answer("Returns", self.index, retrieve=False)["sources"], [])

    def test_hallucinated_citation_detected(self):
        result = answer("Returns", self.index, ScriptedModel(["30 days [invented]"]))
        self.assertEqual(result["invalid_citations"], ["invented"])

    def test_empty_corpus_and_invalid_chunking(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                Index.build(Path(directory))
            with self.assertRaises(ValueError):
                Index.build(Path(directory), chunk_words=10, overlap=10)


class EvaluationTests(unittest.TestCase):
    def test_demos_pass_and_regressions_are_detected(self):
        report = evaluate(runs=2)
        self.assertEqual(report["metrics"]["agent_success_rate"], 1)
        self.assertEqual(report["metrics"]["rag_accuracy"], 1)
        self.assertEqual(len(report["rag_cases"]), 10)
        self.assertEqual(regressions(report, report), [])
        broken = json.loads(json.dumps(report))
        broken["metrics"]["rag_accuracy"] = 0.5
        self.assertEqual(regressions(broken, report), ["rag_accuracy"])

    def test_completion_claim_without_artifact_fails_evaluation(self):
        with patch("ai_projects.evaluate.Agent.run", return_value=AgentResult(True, "done")):
            report = evaluate()
        self.assertEqual(report["metrics"]["agent_success_rate"], 0)

    def test_missing_key_is_actionable(self):
        with patch.dict("os.environ", {}, clear=True):
            for provider in ("openai", "anthropic"):
                with self.assertRaisesRegex(ProviderError, "Set .*_API_KEY"):
                    APIModel(provider).complete("system", [])


if __name__ == "__main__":
    unittest.main()
