import json
import math
import re
from collections import Counter
from pathlib import Path


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def vector(text, idf):
    counts = Counter(tokens(text))
    weighted = {term: count * idf[term] for term, count in counts.items() if term in idf}
    norm = math.sqrt(sum(v * v for v in weighted.values()))
    return {term: weight / norm for term, weight in weighted.items()} if norm else {}


class Index:
    def __init__(self, chunks):
        self.chunks = chunks
        frequencies = Counter(term for chunk in chunks for term in set(tokens(chunk["text"])))
        self.idf = {term: math.log((1 + len(chunks)) / (1 + count)) + 1 for term, count in frequencies.items()}
        self.vectors = [vector(chunk["text"], self.idf) for chunk in chunks]

    @classmethod
    def build(cls, directory: Path, chunk_words=100, overlap=20):
        if chunk_words <= 0 or not 0 <= overlap < chunk_words:
            raise ValueError("Require chunk_words > overlap >= 0")
        chunks = []
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() not in {".txt", ".md"} or not path.is_file():
                continue
            if not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Knowledge files must stay inside the corpus directory")
            words = path.read_text().split()
            for start in range(0, len(words), chunk_words - overlap):
                chunks.append({"id": f"{path.relative_to(directory).as_posix()}:{start}", "source": path.relative_to(directory).as_posix(), "text": " ".join(words[start:start + chunk_words])})
                if start + chunk_words >= len(words):
                    break
        if not chunks:
            raise ValueError("Corpus has no nonempty .txt or .md documents")
        return cls(chunks)

    def search(self, question, k=3):
        if k < 1:
            raise ValueError("k must be positive")
        query = vector(question, self.idf)
        scored = [(sum(weight * vec.get(term, 0) for term, weight in query.items()), chunk) for chunk, vec in zip(self.chunks, self.vectors)]
        scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
        return [{**chunk, "score": round(score, 6)} for score, chunk in scored[:k] if score > 0]

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": 1, "chunks": self.chunks}, indent=2))

    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text())
        if data.get("version") != 1:
            raise ValueError("Unsupported index version")
        return cls(data["chunks"])
