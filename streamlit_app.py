"""Run with: python -m streamlit run streamlit_app.py"""
import hashlib
from pathlib import Path

import streamlit as st

from ai_projects.providers import ProviderError
from ai_projects.task_agent.inventory import parse_inventory
from ai_projects.workspace.services import (
    URL, ask_documents, build_documents, connection_status, run_inventory, upload_signature,
)

st.set_page_config(page_title='CloudShop AI Workspace', page_icon='☁️', layout='wide')
st.markdown('''<style>
.block-container {max-width:1200px; padding-top:2.5rem; padding-bottom:3rem;}
[data-testid="stSidebar"] {background:#f0f4f8;}
[data-testid="stMetric"] {border:1px solid #d8e1eb; border-radius:14px; padding:16px;}
h1 {letter-spacing:-0.045em;} h2 {letter-spacing:-0.025em;}
</style>''', unsafe_allow_html=True)


def sources_view(result):
    if result['sources']:
        with st.expander('Sources & retrieval scores', expanded=True):
            for hit in result['sources']:
                st.caption(f"{hit['id']}  •  Cosine similarity {hit['score']:.3f}")
                st.text(hit['text'])
    if not result.get('supported', False):
        st.caption('No supported answer was found. Try a more specific question or add a relevant document.')


def rag_page(ready):
    st.subheader('RAG Assistant')
    st.write('Turn your documents into answers you can trace back to the source.')
    left, right = st.columns([1, 2], gap='large')
    with left:
        with st.container(border=True):
            st.markdown('#### Knowledge library')
            uploads = st.file_uploader('Upload documents', type=['txt', 'md', 'pdf'], accept_multiple_files=True, key='rag_uploads')
            st.caption('Up to 10 files · 5 MB per file · text-based PDFs; no OCR')
            files = [(upload.name, upload.getvalue()) for upload in uploads]
            signature = upload_signature(files)
            if st.session_state.get('rag_signature') != signature:
                st.session_state.rag_signature = signature
                st.session_state.pop('rag_index', None)
                st.session_state.rag_messages = []
            if st.button('Build knowledge library', type='primary', disabled=not files or not ready, width='stretch'):
                try:
                    with st.spinner('Extracting text and creating local embeddings…'):
                        st.session_state.rag_index = build_documents(files)
                    st.session_state.rag_messages = []
                    st.success('Your library is ready.')
                except (ValueError, OSError, ProviderError) as exc:
                    st.session_state.pop('rag_index', None)
                    st.error(str(exc))
            index = st.session_state.get('rag_index')
            if index:
                st.metric('Indexed chunks', len(index.chunks))
            threshold = st.slider('Minimum retrieval score', 0.0, 1.0, 0.3, 0.05, help='Higher values reject less relevant matches. Scores are similarities, not probabilities.')
            st.caption('Answers use llama3.2. Search uses nomic-embed-text.')
            if st.button('Clear conversation', width='stretch'):
                st.session_state.rag_messages = []
    with right:
        with st.container(border=True):
            st.markdown('#### Ask your documents')
            if not st.session_state.get('rag_index'):
                st.info('Upload documents and build your knowledge library to begin.')
            for message in st.session_state.get('rag_messages', []):
                with st.chat_message(message['role']):
                    st.markdown(message['content'])
                    if 'result' in message:
                        sources_view(message['result'])
            question = st.chat_input('Ask a question about your documents', disabled=not ready or not st.session_state.get('rag_index'), max_chars=4000)
            if question:
                st.session_state.rag_messages.append({'role': 'user', 'content': question})
                with st.chat_message('user'):
                    st.write(question)
                try:
                    with st.spinner('Retrieving sources and generating an answer…'):
                        result = ask_documents(question, st.session_state.rag_index, threshold)
                    st.session_state.rag_messages.append({'role': 'assistant', 'content': result['answer'], 'result': result})
                    with st.chat_message('assistant'):
                        st.markdown(result['answer'])
                        sources_view(result)
                except (ValueError, OSError, ProviderError) as exc:
                    st.error(str(exc))
            st.caption('Each question is answered independently from the current library. Verify important answers against the sources.')


def agent_page(ready):
    st.subheader('AI Agent')
    st.write('Give your inventory a task. Follow each tool action and download a verified reorder report.')
    left, right = st.columns([1, 1.4], gap='large')
    with left:
        with st.container(border=True):
            st.markdown('#### Task setup')
            upload = st.file_uploader('Inventory CSV', type=['csv'], key='agent_upload')
            st.caption('Required columns: product, stock, reorder_level · maximum 64 KB')
            sample = (Path(__file__).parent / 'examples/agent/inventory.csv').read_bytes()
            st.download_button('Download sample CSV', sample, 'inventory.csv', 'text/csv')
            data = upload.getvalue() if upload else b''
            signature = hashlib.sha256(data).hexdigest()
            if st.session_state.get('agent_signature') != signature:
                st.session_state.agent_signature = signature
                st.session_state.pop('agent_output', None)
            valid = False
            if data:
                try:
                    if len(data) > 64_000:
                        raise ValueError('CSV exceeds 64 KB.')
                    items = parse_inventory(data.decode('utf-8-sig'))
                    st.dataframe([{'Product': i.product, 'Stock': i.stock, 'Reorder level': i.reorder_level, 'Order quantity': i.order_quantity} for i in items], hide_index=True, width='stretch')
                    valid = True
                except (ValueError, UnicodeDecodeError) as exc:
                    st.error(str(exc))
            task = st.text_area('What should the agent do?', value='Analyze inventory.csv and create a reorder report for all low-stock products.', max_chars=4000, height=130)
            st.caption('Reorder quantity = max(0, reorder level − stock). Inputs stay read-only. Each run uses an isolated workspace.')
            if st.button('Run inventory agent', type='primary', disabled=not ready or not valid or not task.strip(), width='stretch'):
                st.session_state.pop('agent_output', None)
                try:
                    with st.spinner('Agent is running locally. This may take several model turns…'):
                        st.session_state.agent_output = run_inventory(data, task)
                except (ValueError, OSError, ProviderError) as exc:
                    st.error(str(exc))
    with right:
        output = st.session_state.get('agent_output')
        if not output:
            with st.container(border=True):
                st.markdown('#### Execution workspace')
                st.info('Upload a CSV and run a task to see tool steps and your report here.')
            return
        result = output['result']
        if result['completed'] and output['validated']:
            st.success('Agent finished. Reorder calculations verified.')
        else:
            st.warning('The agent did not complete a verified report task. Review the execution details below.')
        cols = st.columns(3)
        cols[0].metric('Products', output['products'])
        cols[1].metric('Need reorder', output['reorder_count'])
        cols[2].metric('Units to order', output['units_to_order'])
        report_tab, steps_tab = st.tabs(['Inventory report', 'Tool execution'])
        with report_tab:
            if output['validated']:
                st.markdown(output['report'])
                st.download_button('Download report', output['report'], 'inventory-report.md', 'text/markdown', type='primary')
            else:
                st.info('No validated report was generated. Retry the task; the app will not present invented quantities.')
            st.caption(result['answer'])
        with steps_tab:
            if not result['trace']:
                st.info('No tools were executed.')
            for step in result['trace']:
                label = f"Step {step['step']} · {step.get('tool', 'Invalid action')} · {'OK' if step['ok'] else 'Error'}"
                with st.expander(label, expanded=not step['ok']):
                    if not step['ok']:
                        st.error(step.get('message', step.get('error', 'Tool failed')))
                    st.json(step)


with st.sidebar:
    st.markdown('### ☁️ CloudShop')
    st.caption('Your local AI workspace')
    page = st.radio('Workspace', ['RAG Assistant', 'AI Agent'], label_visibility='collapsed', key='workspace_page')
    st.divider()
    if st.button('Refresh Ollama status', width='stretch') or 'ollama_status' not in st.session_state:
        st.session_state.ollama_status = connection_status()
    status = st.session_state.ollama_status
    if status['ready']:
        st.success('Ollama connected · models ready')
    elif status['connected']:
        st.warning('Ollama connected · models missing')
        for name in status['missing']:
            st.code(f'ollama pull {name}')
    else:
        st.error('Ollama is offline')
        st.caption('Start Ollama on this machine, then refresh.')
        st.code('ollama serve')
    st.caption(URL)
    st.caption('llama3.2 · nomic-embed-text')
    st.divider()
    st.caption('Local models only. Documents and inventory are processed on this machine. Temporary files are removed after processing; session results remain until cleared or the session ends.')

st.title('CloudShop AI Workspace')
st.caption('LOCAL INTELLIGENCE  /  DOCUMENTS & OPERATIONS')
if page == 'RAG Assistant':
    rag_page(status['ready'])
else:
    agent_page(status['connected'] and 'llama3.2' not in status['missing'])
