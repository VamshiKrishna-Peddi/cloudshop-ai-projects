import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from ai_projects.providers import APIModel, ProviderError
from ai_projects.task_agent.tools import NoRedirect


class ProviderTests(unittest.TestCase):
    def test_openai_request_and_response_contract(self):
        data = {"choices": [{"message": {"content": '{"final":"done"}'}}]}
        with patch.dict("os.environ", {"OPENAI_API_KEY": "unit-test-placeholder"}), patch("ai_projects.providers.urlopen", return_value=io.BytesIO(json.dumps(data).encode())) as send:
            response = APIModel("openai", "test-model").complete("system", [{"role": "user", "content": "task"}])
        self.assertEqual(response, '{"final":"done"}')
        request = send.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "system"})
        self.assertTrue(send.call_args.kwargs["context"].check_hostname)

    def test_anthropic_request_and_response_contract(self):
        data = {"content": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]}
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "unit-test-placeholder"}), patch("ai_projects.providers.urlopen", return_value=io.BytesIO(json.dumps(data).encode())) as send:
            response = APIModel("anthropic", "test-model").complete("system", [{"role": "user", "content": "task"}])
        self.assertEqual(response, "first\nsecond")
        payload = json.loads(send.call_args.args[0].data)
        self.assertEqual(payload["system"], "system")
        self.assertEqual(payload["max_tokens"], 2048)
        self.assertEqual(send.call_args.args[0].full_url, "https://api.anthropic.com/v1/messages")

    def test_http_errors_hide_response_body(self):
        error = HTTPError("https://api.openai.com", 401, "sensitive details", {}, io.BytesIO(b"secret"))
        with patch.dict("os.environ", {"OPENAI_API_KEY": "unit-test-placeholder"}), patch("ai_projects.providers.urlopen", side_effect=error):
            with self.assertRaisesRegex(ProviderError, "^openai returned HTTP 401$"):
                APIModel("openai").complete("system", [])

    def test_malformed_api_response_rejected(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "unit-test-placeholder"}), patch("ai_projects.providers.urlopen", return_value=io.BytesIO(b"{}")):
            with self.assertRaisesRegex(ProviderError, "unexpected response"):
                APIModel("openai").complete("system", [])

    def test_api_tool_redirect_is_rejected(self):
        with self.assertRaises(ValueError):
            NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://unexpected.example")
