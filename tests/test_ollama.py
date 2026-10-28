import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from ai_projects.ollama import OllamaClient, OllamaModel, OllamaEmbeddings
from ai_projects.providers import ProviderError
from ai_projects.rag.semantic import SemanticIndex
from ai_projects.task_agent.agent import Agent
from ai_projects.task_agent.tools import Tools
from ai_projects.__main__ import main
from ai_projects.evaluate import regressions


class OllamaTests(unittest.TestCase):
    def test_chat_and_json_action_contract(self):
        model = OllamaModel()
        with patch.object(model.client, 'post', return_value={'message': {'content': 'answer'}}) as post:
            self.assertEqual(model.complete('system', [{'role': 'user', 'content': 'question'}]), 'answer')
            payload = post.call_args.args[1]
            self.assertEqual(payload['model'], 'llama3.2')
            self.assertFalse(payload['stream'])
            self.assertNotIn('format', payload)
            model.complete_action('system', [])
            self.assertEqual(post.call_args.args[1]['format'], 'json')

    def test_transport_and_failure_messages(self):
        client = OllamaClient('http://127.0.0.1:11434')
        with patch.object(client.opener, 'open', return_value=io.BytesIO(b'{"ok":true}')) as send:
            self.assertEqual(client.post('/api/chat', {}), {'ok': True})
            self.assertEqual(send.call_args.args[0].full_url, 'http://127.0.0.1:11434/api/chat')
        with patch.object(client.opener, 'open', side_effect=URLError('refused')):
            with self.assertRaisesRegex(ProviderError, 'ollama serve'):
                client.post('/api/chat', {})
        with patch.object(client.opener, 'open', side_effect=HTTPError('url', 404, 'missing', {}, None)):
            with self.assertRaisesRegex(ProviderError, 'ollama pull'):
                client.post('/api/chat', {})

    def test_local_addresses_only(self):
        for url in ['https://api.openai.com', 'http://remote.example:11434', 'http://localhost:11434/path', 'http://user@localhost:11434']:
            with self.assertRaises(ValueError):
                OllamaClient(url)

    def test_embeddings_normalized_and_invalid_responses_rejected(self):
        embedder = OllamaEmbeddings()
        with patch.object(embedder.client, 'post', return_value={'embeddings': [[3, 4]]}) as post:
            self.assertEqual(embedder.embed(['document']), [[0.6, 0.8]])
            self.assertEqual(post.call_args.args, ('/api/embed', {'model': 'nomic-embed-text', 'input': ['document'], 'truncate': False}))
        for value in [[], [[0, 0]], [[float('nan')]], [['bad']], [[True]], [[1], [1]]]:
            with patch.object(embedder.client, 'post', return_value={'embeddings': value}):
                with self.assertRaises(ProviderError):
                    embedder.embed(['text'])

    def test_agent_executes_ollama_json_tool_actions(self):
        model = OllamaModel()
        responses = [{'message': {'content': json.dumps(action)}} for action in [
            {'tool': 'write_file', 'args': {'path': 'result.txt', 'content': 'local model result'}},
            {'tool': 'read_file', 'args': {'path': 'result.txt'}},
            {'final': 'done'},
        ]]
        with tempfile.TemporaryDirectory() as directory, patch.object(model.client, 'post', side_effect=responses) as post:
            result = Agent(model, Tools(Path(directory))).run('Write and read result.txt')
            self.assertTrue(result.completed)
            self.assertEqual(result.trace[1]['result'], 'local model result')
            self.assertTrue(all(call.args[1]['format'] == 'json' for call in post.call_args_list))
            self.assertIn('Tool observation:', post.call_args_list[1].args[1]['messages'][-1]['content'])

    def test_semantic_search_persistence_and_prefixes(self):
        embedder = OllamaEmbeddings()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'a.md').write_text('Returns are accepted.')
            (root / 'b.md').write_text('Shipping takes five days.')
            with patch.object(embedder, 'embed', return_value=[[1., 0.], [0., 1.]]) as embed:
                index = SemanticIndex.build(root, embedder)
                self.assertTrue(embed.call_args.args[0][0].startswith('search_document: '))
            path = root / 'index.json'
            index.save(path)
            loaded = SemanticIndex.load(path, embedder)
            with patch.object(embedder, 'embed', return_value=[[0., 1.]]) as embed:
                self.assertEqual(loaded.search('delivery')[0]['source'], 'b.md')
                self.assertEqual(embed.call_args.args[0], ['search_query: delivery'])
            with self.assertRaises(ValueError):
                SemanticIndex.load(path, OllamaEmbeddings('another-model'))
            with patch.object(embedder, 'embed', return_value=[[1., 0., 0.]]):
                with self.assertRaisesRegex(ValueError, 'dimension'):
                    loaded.search('query')

    def test_cli_routes_rag_to_both_local_models(self):
        with patch('ai_projects.ollama.OllamaClient.post', side_effect=[
            {'embeddings': [[1., 0.], [0., 1.], [1., 0.]]},
            {'embeddings': [[1., 0.]]},
            {'message': {'content': '5 business days [shipping.md:0]'}},
        ]) as post, patch('sys.stdout', new_callable=io.StringIO) as output:
            self.assertEqual(main(['rag', 'shipping', '--provider', 'ollama']), 0)
            result = json.loads(output.getvalue())
            self.assertIn('5 business days', result['answer'])
            self.assertEqual([call.args[1]['model'] for call in post.call_args_list], ['nomic-embed-text', 'nomic-embed-text', 'llama3.2'])

    def test_bad_chat_and_embedding_baseline_mismatch(self):
        model = OllamaModel()
        with patch.object(model.client, 'post', return_value={'message': {'content': ''}}):
            with self.assertRaises(ProviderError):
                model.complete('system', [])
        report = {'mode': 'local-ollama', 'provider': 'ollama', 'model': 'llama3.2', 'embedding_model': 'nomic-embed-text'}
        with self.assertRaisesRegex(ValueError, 'embedding model'):
            regressions(report, {**report, 'embedding_model': 'different'})
