"""Model-local query history and reversible relevance feedback."""
from dataclasses import dataclass, field
from .query_refinement import query_text
from .relevance_feedback import refine_vector


@dataclass
class RetrievalSession:
    history: list = field(default_factory=list)
    positive: set = field(default_factory=set)
    negative: set = field(default_factory=set)

    def add_query(self,text):
        if not isinstance(text,str) or not text.strip():
            raise ValueError('Query must be a nonempty string.')
        self.history.append(text.strip())

    def query_text(self,mode='accumulated'):
        return query_text(self.history,mode)

    def set_feedback(self,item_id,relevant):
        self.positive.discard(item_id)
        self.negative.discard(item_id)
        (self.positive if relevant else self.negative).add(item_id)

    def refine(self,query,gallery,beta=.5,gamma=.25):
        return refine_vector(query,gallery.vectors(sorted(self.positive)),gallery.vectors(sorted(self.negative)),beta,gamma)

    def reset(self):
        self.history.clear();self.positive.clear();self.negative.clear()
