"""
Hybrid Memory Manager — Generative Agent Layer

Combines a short-term sliding window (deque) with a long-term FAISS
semantic index, so the agent can recall the outcomes of earlier
sub-tasks during multi-step instructions (see dissertation Ch. 9.5).
"""

import faiss
import numpy as np
from collections import deque
from sentence_transformers import SentenceTransformer


class HybridMemoryManager:
    """
    short_term : deque(maxlen=window_size)
        Rolling window of the most recent interaction steps.
    index : faiss.IndexFlatL2
        Exact nearest-neighbour semantic memory over summarised
        sub-task history, queried at each planning step.
    """

    def __init__(self, window_size=10, embed_dim=384):
        self.short_term = deque(maxlen=window_size)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatL2(embed_dim)
        self.summaries = []
        self.completed_count = 0

    def add_step(self, step: dict):
        self.short_term.append(step)
        if step.get('status') == 'completed':
            self.completed_count += 1
            if self.completed_count % 5 == 0:
                self._summarise_and_store()

    def get_context(self, current_goal: str) -> dict:
        recent = list(self.short_term)
        relevant = self._retrieve_relevant(current_goal)
        return {'recent_steps': recent, 'relevant_summaries': relevant}

    def _summarise_and_store(self):
        steps_text = ' '.join([str(s) for s in list(self.short_term)[-5:]])
        summary = f'Completed steps summary: {steps_text[:200]}'
        embedding = self.encoder.encode([summary])
        self.index.add(np.array(embedding, dtype=np.float32))
        self.summaries.append(summary)

    def _retrieve_relevant(self, query: str, k=3) -> list:
        if self.index.ntotal == 0:
            return []
        query_embed = self.encoder.encode([query])
        distances, indices = self.index.search(
            np.array(query_embed, dtype=np.float32), min(k, self.index.ntotal))
        return [self.summaries[i] for i in indices[0] if i < len(self.summaries)]
