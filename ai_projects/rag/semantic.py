"""Persistent dense-vector cosine search using local Ollama embeddings."""
import json
import math
from pathlib import Path

from .retrieval import Index


class SemanticIndex:
    def __init__(self, chunks, vectors, embedder):
        if not chunks or len(chunks) != len(vectors):
            raise ValueError("Index chunks and vectors must be nonempty and have equal length")
        dimension = len(vectors[0])
        for row in vectors:
            if (not dimension or len(row) != dimension
                    or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in row)
                    or not math.isclose(sum(x*x for x in row), 1, rel_tol=1e-5, abs_tol=1e-5)):
                raise ValueError("Index vectors must be finite, normalized and have equal dimensions")
        self.chunks, self.vectors, self.embedder = chunks, vectors, embedder

    @classmethod
    def build(cls, directory, embedder):
        chunks = Index.build(directory).chunks
        # Nomic recommends distinct prefixes for documents and search queries.
        vectors = []
        for offset in range(0, len(chunks), 32):
            vectors.extend(embedder.embed(["search_document: " + c["text"] for c in chunks[offset:offset + 32]]))
        return cls(chunks, vectors, embedder)

    def search(self, question, k=3):
        if k < 1:
            raise ValueError("k must be positive")
        query = self.embedder.embed(["search_query: " + question])[0]
        if len(query) != len(self.vectors[0]):
            raise ValueError("Embedding dimension changed; rebuild the index")
        scored = [(sum(x*y for x, y in zip(query, row)), chunk) for row, chunk in zip(self.vectors, self.chunks)]
        scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
        return [{**chunk, "score": round(score, 6)} for score, chunk in scored[:k] if score > 0]

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": 2, "kind": "ollama", "embedding_model": self.embedder.model, "chunks": self.chunks, "vectors": self.vectors}, indent=2))

    @classmethod
    def load(cls, path, embedder):
        data = json.loads(path.read_text())
        if data.get("version") != 2 or data.get("kind") != "ollama" or data.get("embedding_model") != embedder.model:
            raise ValueError("Index type or embedding model mismatch; rebuild with the selected embedding model")
        return cls(data["chunks"], data["vectors"], embedder)
