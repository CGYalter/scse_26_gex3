"""RAG retrieval engine.

Loads the university IT policy documents (the single knowledge-base file used in
this graded exercise) and exposes a lightweight TF-IDF retriever that returns the
policy excerpts most relevant to a user's question.

The retrieval here is intentionally self-contained (pure Python + NumPy) so that
it runs anywhere without an external vector store. The document is loaded ONLY
through ``load_policy_documents()``, which is provided and must not be changed.
"""

import math
import re

import numpy as np

from policy_loader import load_policy_documents

# The knowledge base for this exercise is a single file of policy sections.
documents = load_policy_documents()

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text):
    """Lowercase and split text into alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


class TfidfRetriever:
    """Classic TF-IDF cosine-similarity retriever over a small document set."""

    def __init__(self, documents, k=3):
        self.documents = list(documents)
        self.k = k
        self._build_index()

    def _build_index(self):
        vocab = {}
        doc_tokens = []
        for doc in self.documents:
            tokens = _tokenize(doc.page_content)
            doc_tokens.append(tokens)
            for token in set(tokens):
                vocab[token] = vocab.get(token, 0) + 1

        self.word2idx = {word: i for i, word in enumerate(vocab)}
        n_docs = len(self.documents)
        n_terms = len(self.word2idx)

        # Inverse document frequency, smoothed.
        idf = np.array(
            [
                math.log((1.0 + n_docs) / (1.0 + vocab[word])) + 1.0
                for word in self.word2idx
            ]
        )

        # Term-frequency matrix.
        tf = np.zeros((n_docs, n_terms))
        for i, tokens in enumerate(doc_tokens):
            if not tokens:
                continue
            for word in set(tokens):
                tf[i, self.word2idx[word]] = tokens.count(word) / len(tokens)

        matrix = tf * idf
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._doc_matrix = matrix / norms
        self._idf = idf

    def retrieve(self, query, k=None):
        """Return up to ``k`` most relevant Document objects for ``query``."""
        k = k or self.k
        tokens = _tokenize(query)

        query_vec = np.zeros(len(self.word2idx))
        if tokens:
            for word in set(tokens):
                if word in self.word2idx:
                    query_vec[self.word2idx[word]] = tokens.count(word) / len(tokens)

        query_vec = query_vec * self._idf
        norm = np.linalg.norm(query_vec)
        if norm == 0:
            return []
        query_vec = query_vec / norm

        scores = self._doc_matrix @ query_vec
        order = np.argsort(-scores)
        return [
            self.documents[i]
            for i in order[:k]
            if scores[i] > 0
        ]


retriever = TfidfRetriever(documents, k=3)


def retrieve(query, k=3):
    """Convenience wrapper around the module-level retriever."""
    return retriever.retrieve(query, k=k)
