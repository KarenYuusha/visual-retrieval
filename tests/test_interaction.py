import numpy as np
import pytest
from visual_retrieval.retrieval.index import GalleryIndex
from visual_retrieval.interaction.session import RetrievalSession
from visual_retrieval.interaction.relevance_feedback import refine_vector


def test_session_accumulates_queries_and_feedback_changes_rank():
    gallery=GalleryIndex(['a','b'],np.eye(2))
    session=RetrievalSession()
    session.add_query('a person');session.add_query('wearing blue')
    assert session.query_text('latest')=='wearing blue'
    assert session.query_text('accumulated')=='a person. wearing blue'
    query=np.array([1.,0.])
    assert gallery.search(query,1)[0]['item_id']=='a'
    session.set_feedback('b',True);session.set_feedback('a',False)
    refined=session.refine(query,gallery,beta=2,gamma=1)
    assert gallery.search(refined,1)[0]['item_id']=='b'
    session.set_feedback('b',False)
    assert 'b' not in session.positive


def test_feedback_validates_shapes_and_zero_vectors():
    with pytest.raises(ValueError):
        refine_vector(np.zeros(2))
    with pytest.raises(ValueError):
        refine_vector(np.ones(2),positive=np.ones((1,3)))
    with pytest.raises(ValueError):
        refine_vector(np.ones(2),beta=-1)


def test_session_feedback_applies_on_next_turn():
    from visual_retrieval.evaluation.session_evaluation import evaluate_sessions
    class Encoder:
        def encode_text(self,texts):
            return np.array([[1.,0.]])
    sessions=[dict(target_item_id='b',turns=[dict(query='person',positive=['b'],negative=['a']),dict(query='blue shirt')])]
    report=evaluate_sessions(sessions,GalleryIndex(['a','b'],np.eye(2)),Encoder(),'feedback',beta=2,gamma=1)
    assert [t['rank'] for t in report['sessions'][0]['turns']]==[2,1]
    assert report['by_turn']['2']['r1']==100
