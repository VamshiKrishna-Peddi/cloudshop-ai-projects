import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from ai_projects.rag.retrieval import Index
from test_workspace import CSV, Model, agent_model

APP = str(Path(__file__).resolve().parents[1] / 'streamlit_app.py')
READY = {'connected':True, 'ready':True, 'missing':[]}


class BrowserUITests(unittest.TestCase):
    def test_offline_dashboard_and_page_navigation(self):
        with patch('ai_projects.workspace.services.connection_status', return_value={'connected':False,'ready':False,'missing':[]}):
            app = AppTest.from_file(APP).run()
            self.assertEqual(len(app.exception),0)
            self.assertEqual(app.title[0].value,'CloudShop AI Workspace')
            self.assertIn('Ollama is offline',[error.value for error in app.error])
            app.radio[0].set_value('AI Agent').run()
            self.assertEqual(len(app.exception),0)
            self.assertTrue(next(button for button in app.button if button.label=='Run inventory agent').disabled)

    def test_inventory_upload_execution_report_and_download(self):
        upload = SimpleNamespace(name='inventory.csv',getvalue=lambda:CSV)
        with patch('ai_projects.workspace.services.connection_status',return_value=READY), patch('streamlit.file_uploader',return_value=upload), patch('ai_projects.workspace.services.OllamaModel',return_value=agent_model()):
            app = AppTest.from_file(APP)
            app.session_state['ollama_status']=READY
            app.session_state['workspace_page']='AI Agent'
            app.run()
            self.assertEqual(len(app.exception),0)
            next(button for button in app.button if button.label=='Run inventory agent').click().run()
            self.assertEqual(len(app.exception),0)
            self.assertTrue(app.session_state['agent_output']['validated'])
            self.assertIn('Keyboard: stock 3; order 7',app.session_state['agent_output']['report'])
            self.assertEqual(len(app.get('download_button')),2)

    def test_rag_upload_build_chat_sources_and_change_invalidation(self):
        upload = SimpleNamespace(name='policy.md',getvalue=lambda:b'Returns within 30 days.')
        index = Index([{'id':'policy:0','source':'policy','text':'Returns within 30 days.'}])
        with patch('ai_projects.workspace.services.connection_status',return_value=READY), patch('streamlit.file_uploader',return_value=[upload]), patch('ai_projects.workspace.services.SemanticIndex.build',return_value=index), patch('ai_projects.workspace.services.OllamaModel',return_value=Model(['30 days [policy:0]'])):
            app = AppTest.from_file(APP).run()
            next(button for button in app.button if button.label=='Build knowledge library').click().run()
            self.assertEqual(len(app.exception),0)
            app.chat_input[0].set_value('Returns').run()
            self.assertEqual(len(app.exception),0)
            self.assertEqual(len(app.session_state['rag_messages']),2)
            self.assertTrue(app.session_state['rag_messages'][1]['result']['supported'])
            self.assertTrue(any('Cosine similarity' in caption.value for caption in app.caption))
            upload.getvalue=lambda:b'Changed document'
            app.run()
            self.assertNotIn('rag_index',app.session_state)
            self.assertEqual(app.session_state['rag_messages'],[])
