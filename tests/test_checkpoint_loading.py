"""Restricted loading supports ordinary metadata but rejects custom globals."""
import pickle
import pytest
import torch
from visual_retrieval.models.checkpoint import load_checkpoint


class UnapprovedMetadata:
    pass


def test_checkpoint_set_metadata_loads_without_unrestricted_pickle(tmp_path):
    path=tmp_path/'checkpoint.pt'
    weight=torch.tensor([1.,2.])
    torch.save({'model':{'vision_proj.weight':weight}, 'metadata':{'tags':{'video','stage2'}}},path)
    before=list(torch.serialization.get_safe_globals())
    loaded=load_checkpoint(path)
    torch.testing.assert_close(loaded['model']['vision_proj.weight'],weight)
    assert loaded['metadata']['tags']=={'video','stage2'}
    assert torch.serialization.get_safe_globals()==before


def test_custom_checkpoint_globals_remain_rejected(tmp_path):
    path=tmp_path/'unapproved.pt'
    torch.save({'model':{'weight':torch.ones(2)},'metadata':UnapprovedMetadata()},path)
    before=list(torch.serialization.get_safe_globals())
    with pytest.raises(pickle.UnpicklingError,match='Unsupported global'):
        load_checkpoint(path)
    assert torch.serialization.get_safe_globals()==before


def test_preserves_callers_existing_set_allowlist(tmp_path):
    path=tmp_path/'existing.pt'
    torch.save({'tags':{'stage2'}},path)
    with torch.serialization.safe_globals([set]):
        assert load_checkpoint(path)['tags']=={'stage2'}
        assert set in torch.serialization.get_safe_globals()
