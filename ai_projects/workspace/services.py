import hashlib
import io
import json
import tempfile
from pathlib import Path
from urllib.request import Request

from ai_projects.ollama import OllamaClient, OllamaEmbeddings, OllamaModel
from ai_projects.providers import ProviderError
from ai_projects.rag.pipeline import UNKNOWN, answer
from ai_projects.rag.semantic import SemanticIndex
from ai_projects.task_agent.agent import Agent, AgentResult
from ai_projects.task_agent.inventory import parse_inventory, render_report
from ai_projects.task_agent.tools import Tools

URL = 'http://localhost:11434'


def connection_status():
    try:
        client = OllamaClient(URL)
        with client.opener.open(Request(URL + '/api/tags'), timeout=2) as response:
            names = [model['name'] for model in json.load(response)['models']]
        missing = [name for name in ('llama3.2', 'nomic-embed-text') if name not in names and name + ':latest' not in names]
        return {'connected': True, 'ready': not missing, 'missing': missing}
    except (OSError, ValueError, KeyError, TypeError):
        return {'connected': False, 'ready': False, 'missing': []}


def upload_signature(files):
    digest = hashlib.sha256()
    for name, data in files:
        digest.update(json.dumps([name, len(data)]).encode())
        digest.update(data)
    return digest.hexdigest()


def extract_document(name, data):
    if not data or len(data) > 5_000_000:
        raise ValueError('Each document must be nonempty and no larger than 5 MB.')
    suffix = Path(name).suffix.lower()
    if suffix in {'.txt', '.md'}:
        try:
            text = data.decode('utf-8-sig')
        except UnicodeDecodeError:
            raise ValueError('Text and Markdown documents must use UTF-8 encoding.') from None
    elif suffix == '.pdf':
        from pypdf import PdfReader
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise ValueError('Encrypted PDFs are unsupported; upload an unlocked copy.')
            if len(reader.pages) > 100:
                raise ValueError('PDFs must have at most 100 pages.')
            text = '\n\n'.join(page.extract_text() or '' for page in reader.pages)
        except ValueError:
            raise
        except Exception:
            raise ValueError('Could not extract this PDF. Upload a readable, unlocked PDF.') from None
    else:
        raise ValueError('Supported document types are TXT, Markdown (.md), and PDF.')
    if not text.strip():
        raise ValueError('No readable text found. Scanned PDFs require OCR before upload.')
    if len(text) > 500_000:
        raise ValueError('Extracted document exceeds 500,000 characters; split it into smaller files.')
    return text


def build_documents(files, embedder=None):
    if not 1 <= len(files) <= 10:
        raise ValueError('Upload between 1 and 10 documents.')
    documents = []
    total = 0
    for number, (name, data) in enumerate(files, 1):
        text = extract_document(name, data)
        total += len(text)
        if total > 1_000_000:
            raise ValueError('Combined extracted text exceeds 1,000,000 characters.')
        # Preserve a readable source name while preventing traversal and collisions.
        safe_name = Path(name.replace('\\', '/')).name
        safe_name = ''.join(c for c in safe_name if c.isalnum() or c in ' ._-')[:100] or 'document'
        documents.append((f'{number:02d}-{safe_name}.txt', text))
    with tempfile.TemporaryDirectory(prefix='cloudshop-docs-') as directory:
        for name, text in documents:
            (Path(directory) / name).write_text(text, encoding='utf-8')
        return SemanticIndex.build(Path(directory), embedder or OllamaEmbeddings(base_url=URL))


class FilteredIndex:
    def __init__(self, index, threshold):
        self.index, self.threshold = index, threshold

    def search(self, question, k=3):
        return [hit for hit in self.index.search(question, k) if hit['score'] >= self.threshold]


def ask_documents(question, index, threshold=0.3, k=3, model=None):
    if not question.strip() or len(question) > 4000:
        raise ValueError('Enter a question between 1 and 4,000 characters.')
    if not 0 <= threshold <= 1:
        raise ValueError('Retrieval threshold must be between 0 and 1.')
    filtered = FilteredIndex(index, threshold)
    # Do not ask a model to guess if retrieval found no adequate context.
    hits = filtered.search(question, k)
    if not hits:
        return {'question': question, 'answer': UNKNOWN, 'sources': [], 'citations': [], 'invalid_citations': [], 'supported': False}
    class Retrieved:
        def search(self, question, k=3):
            return hits[:k]
    result = answer(question, Retrieved(), model or OllamaModel('llama3.2', URL), k=k)
    text = result['answer'].lower()
    abstained = any(phrase in text for phrase in ("don't know", 'do not know', 'not provided', 'not mentioned', 'insufficient context', 'cannot answer'))
    result['supported'] = bool(result['citations']) and not result['invalid_citations'] and not abstained
    if not result['supported']:
        result['answer'] = UNKNOWN
    return result


class InventoryTools(Tools):
    """Prevent model arithmetic or input edits from corrupting the reorder report."""
    def __init__(self, root, csv_text):
        super().__init__(root)
        self.expected = render_report(parse_inventory(csv_text))
        (self.root / 'inventory.csv').write_text(csv_text, encoding='utf-8')

    def call(self, name, args):
        if name == 'write_file' and isinstance(args, dict):
            path = self.path(args.get('path', ''))
            if path == self.root / 'inventory.csv':
                raise ValueError('The uploaded inventory is read-only.')
            if path == self.root / 'report.md' and args.get('content') != self.expected:
                raise ValueError('Use inventory_report to write report.md with validated reorder calculations.')
        return super().call(name, args)


def run_inventory(data, task, model=None):
    if not data or len(data) > 64_000:
        raise ValueError('Inventory CSV must be nonempty and no larger than 64 KB.')
    if not task.strip() or len(task) > 4000:
        raise ValueError('Enter a task between 1 and 4,000 characters.')
    try:
        text = data.decode('utf-8-sig')
    except UnicodeDecodeError:
        raise ValueError('Inventory CSV must use UTF-8 encoding.') from None
    items = parse_inventory(text)
    with tempfile.TemporaryDirectory(prefix='cloudshop-agent-') as directory:
        tools = InventoryTools(Path(directory), text)
        instruction = task + '\nUse the uploaded inventory.csv. You must call inventory_report with path inventory.csv and output report.md to produce the verified reorder report. Do not change the input.'
        agent = Agent(model or OllamaModel('llama3.2', URL), tools)
        try:
            result = agent.run(instruction)
        except ProviderError as exc:
            # Retain successful steps and artifacts even if a later model request fails.
            result = AgentResult(False, str(exc), getattr(agent, 'trace', []))
        report_path = tools.root / 'report.md'
        report = report_path.read_text() if report_path.is_file() else ''
        valid = report == tools.expected
        return {'result': result.to_dict(), 'report': report if valid else '', 'validated': valid,
                'products': len(items), 'reorder_count': sum(i.order_quantity > 0 for i in items),
                'units_to_order': sum(i.order_quantity for i in items)}
