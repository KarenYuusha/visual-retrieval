import numpy as np
import torch

from iv2_model import InternVideo2Tokenizer, encode_text_batch


def tokenizer_fixture(tmp_path):
    path = tmp_path / 'vocab.txt'
    path.write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]',
                               'a', 'man', 'walks', 'dog', 'runs']) + '\n')
    return InternVideo2Tokenizer(str(path))


def test_single_caption_has_cls_but_no_sep(tmp_path):
    tokenizer = tokenizer_fixture(tmp_path)
    tokens = tokenizer('a man walks', padding='max_length', truncation=True,
                       max_length=8, return_tensors='pt')
    assert tokens.input_ids.tolist() == [[2, 5, 6, 7, 0, 0, 0, 0]]
    assert tokens.attention_mask.tolist() == [[1, 1, 1, 1, 0, 0, 0, 0]]
    assert tokens.token_type_ids.shape == tokens.input_ids.shape
    assert tokenizer.num_special_tokens_to_add(pair=False) == 1


def test_long_caption_keeps_39_wordpieces_after_cls(tmp_path):
    tokenizer = tokenizer_fixture(tmp_path)
    tokens = tokenizer(' '.join(['man'] * 50), max_length=40, truncation=True,
                       padding='max_length', return_tensors='pt')
    assert tokens.input_ids.tolist() == [[2] + [6] * 39]
    assert tokens.attention_mask.sum().item() == 40


def test_single_caption_special_token_mask_matches_actual_length(tmp_path):
    tokenizer = tokenizer_fixture(tmp_path)
    tokens = tokenizer('a man', return_special_tokens_mask=True)
    assert tokens['input_ids'] == [2, 5, 6]
    assert tokens['special_tokens_mask'] == [1, 0, 0]
    assert tokens['token_type_ids'] == [0, 0, 0]


def test_text_batch_uses_corrected_tokens_after_caption_cleaning(tmp_path):
    tokenizer = tokenizer_fixture(tmp_path)

    class CaptureModel:
        def text_features(self, tokens):
            self.tokens = tokens
            return torch.ones(len(tokens.input_ids), 512)

    model = CaptureModel()
    features = encode_text_batch(model, tokenizer, ['A man walks!', 'A dog runs.'], 'cpu')
    assert features.shape == (2, 512)
    assert features.dtype == np.float32
    assert model.tokens.input_ids[:, :5].tolist() == [[2, 5, 6, 7, 0], [2, 5, 8, 9, 0]]
    assert model.tokens.attention_mask.sum(1).tolist() == [4, 4]
