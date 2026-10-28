import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ai_projects.providers import ProviderError
from ai_projects.rag.retrieval import Index
from ai_projects.task_agent.inventory import parse_inventory, render_report
from ai_projects.workspace.services import (
    ask_documents, build_documents, connection_status, extract_document,
    run_inventory, upload_signature,
)

CSV = b'product,stock,reorder_level\nKeyboard,3,10\nMouse,5,5\nMonitor,1,4\nExtra,20,5\n'


class Model:
    def __init__(self, actions):
        self.actions = iter(actions)

    def complete(self, system, messages):
        action = next(self.actions)
        if isinstance(action, Exception):
            raise action
        return action


def agent_model():
    return Model([json.dumps({'tool': 'inventory_report', 'args': {'path': 'inventory.csv', 'output': 'report.md'}}), json.dumps({'final': 'Done'})])


class InventoryTests(unittest.TestCase):
    def test_correct_boundaries_and_report(self):
        items = parse_inventory(CSV.decode())
        self.assertEqual([i.order_quantity for i in items], [7, 0, 3, 0])
        report = render_report(items)
        self.assertIn('Keyboard: stock 3; order 7', report)
        self.assertIn('Monitor: stock 1; order 3', report)
        self.assertNotIn('Mouse', report)
        self.assertIn('Products requiring reorder: 2', report)

    def test_bad_inventory_rejected(self):
        for text in ('product,stock\na,1', 'product,stock,reorder_level\na,-1,4',
                     'product,stock,reorder_level\na,1.5,4', 'product,stock,reorder_level\na,1,4\nA,2,3',
                     'product,stock,reorder_level\na,1', 'product,stock,reorder_level\na,1,4,extra',
                     'product,stock,reorder_level\n,1,4', 'product,stock,reorder_level\n'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_inventory(text)

    def test_existing_agent_runs_deterministic_tool(self):
        result = run_inventory(CSV, 'Create a reorder report', agent_model())
        self.assertTrue(result['validated'])
        self.assertTrue(result['result']['completed'])
        self.assertEqual(result['units_to_order'], 10)
        self.assertEqual(result['result']['trace'][0]['tool'], 'inventory_report')

    def test_bad_model_math_is_rejected_then_recovered(self):
        actions = [json.dumps({'tool':'write_file','args':{'path':'report.md','content':'Keyboard: order 999'}}),
                   json.dumps({'tool':'write_file','args':{'path':'inventory.csv','content':'fake'}}),
                   json.dumps({'tool':'inventory_report','args':{}}), json.dumps({'final':'Done'})]
        result = run_inventory(CSV, 'Create a report', Model(actions))
        self.assertTrue(result['validated'])
        self.assertFalse(result['result']['trace'][0]['ok'])
        self.assertFalse(result['result']['trace'][1]['ok'])
        self.assertNotIn('999', result['report'])

    def test_claim_without_report_not_accepted(self):
        result = run_inventory(CSV, 'Report', Model(['{"final":"Done"}']))
        self.assertFalse(result['validated'])
        self.assertEqual(result['report'], '')

    def test_provider_failure_preserves_trace_and_artifact(self):
        result = run_inventory(CSV, 'Report', Model(['{"tool":"inventory_report","args":{}}', ProviderError('Ollama unavailable')]))
        self.assertFalse(result['result']['completed'])
        self.assertTrue(result['validated'])
        self.assertEqual(len(result['result']['trace']), 1)


class DocumentTests(unittest.TestCase):
    def test_upload_formats_and_invalid_inputs(self):
        self.assertEqual(extract_document('a.md', b'# Shipping\nFive days'), '# Shipping\nFive days')
        for name, data in [('a.exe',b'bad'), ('a.txt',b'\xff'), ('a.txt',b' '), ('a.pdf',b'broken'), ('a.txt',b'x'*5_000_001)]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                extract_document(name,data)

    def test_pdf_text_and_blank_pdf(self):
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
        writer = PdfWriter()
        page = writer.add_blank_page(width=200, height=200)
        font = DictionaryObject({NameObject('/Type'):NameObject('/Font'), NameObject('/Subtype'):NameObject('/Type1'), NameObject('/BaseFont'):NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):font})})
        stream = DecodedStreamObject()
        stream.set_data(b'BT /F1 12 Tf 20 150 Td (Returns within 30 days.) Tj ET')
        page[NameObject('/Contents')] = stream
        buffer = io.BytesIO()
        writer.write(buffer)
        self.assertIn('30 days',extract_document('policy.pdf',buffer.getvalue()))
        blank = PdfWriter()
        blank.add_blank_page(width=200,height=200)
        buffer = io.BytesIO()
        blank.write(buffer)
        with self.assertRaisesRegex(ValueError,'OCR'):
            extract_document('scanned.pdf',buffer.getvalue())
        encrypted = PdfWriter()
        encrypted.add_blank_page(width=200,height=200)
        encrypted.encrypt('password')
        buffer = io.BytesIO()
        encrypted.write(buffer)
        with self.assertRaisesRegex(ValueError,'Encrypted'):
            extract_document('locked.pdf',buffer.getvalue())

    def test_upload_build_reuses_semantic_index_and_preserves_names(self):
        embedder = SimpleNamespace(embed=lambda texts:[[1.,0.] for _ in texts])
        index = build_documents([('../../policy.md',b'Returns in 30 days.'), ('policy.md',b'Shipping in 5 days.')], embedder)
        self.assertEqual(len(index.chunks),2)
        self.assertNotIn('../',index.chunks[0]['source'])
        self.assertNotEqual(index.chunks[0]['source'],index.chunks[1]['source'])
        self.assertEqual(index.search('returns')[0]['score'],1)
        self.assertNotEqual(upload_signature([('a',b'x')]), upload_signature([('a',b'y')]))

    def test_answer_citations_and_abstention(self):
        index = Index([{'id':'policy:0','source':'policy','text':'Returns within 30 days.'}])
        result = ask_documents('Returns',index,model=Model(['30 days [policy:0]']))
        self.assertTrue(result['supported'])
        self.assertEqual(result['sources'][0]['id'],'policy:0')
        for response in ['30 days [made-up]', '30 days', "I don't know [policy:0]"]:
            self.assertFalse(ask_documents('Returns',index,model=Model([response]))['supported'])
        self.assertFalse(ask_documents('xyzzy',index,model=Model([]))['supported'])
        self.assertFalse(ask_documents('Returns',index,threshold=1,model=Model([]))['supported'])

    def test_status_reports_models_and_offline(self):
        with patch('ai_projects.workspace.services.OllamaClient') as client:
            client.return_value.opener.open.return_value = io.BytesIO(b'{"models":[{"name":"llama3.2:latest"},{"name":"nomic-embed-text:latest"}]}')
            self.assertTrue(connection_status()['ready'])
            client.return_value.opener.open.return_value = io.BytesIO(b'{"models":[{"name":"llama3.2:latest"}]}')
            self.assertEqual(connection_status()['missing'],['nomic-embed-text'])
            client.return_value.opener.open.side_effect = OSError('offline')
            self.assertFalse(connection_status()['connected'])
