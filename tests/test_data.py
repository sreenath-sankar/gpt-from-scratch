import tiktoken
import torch

from gpt.data import GPTDatasetV1, create_dataloader_v1

TEXT = "In the beginning was the Word, and the Word was with God, and the Word was God."


def test_target_is_input_shifted_by_one():
    tok = tiktoken.get_encoding("gpt2")
    ds = GPTDatasetV1(TEXT, tok, max_length=4, stride=4)
    inp, tgt = ds[0]
    # Everything but the first input token should match everything but the
    # last target token.
    assert torch.equal(inp[1:], tgt[:-1])


def test_consecutive_windows_overlap_by_max_length_minus_stride():
    tok = tiktoken.get_encoding("gpt2")
    ds = GPTDatasetV1(TEXT, tok, max_length=4, stride=1)
    inp0, _ = ds[0]
    inp1, _ = ds[1]
    # With stride 1 the second window is the first shifted by one token.
    assert torch.equal(inp0[1:], inp1[:-1])


def test_dataset_length_matches_window_count():
    tok = tiktoken.get_encoding("gpt2")
    n_tokens = len(tok.encode(TEXT))
    max_length, stride = 4, 4
    ds = GPTDatasetV1(TEXT, tok, max_length, stride)
    expected = len(range(0, n_tokens - max_length, stride))
    assert len(ds) == expected


def test_dataloader_batch_shapes():
    dl = create_dataloader_v1(TEXT, batch_size=2, max_length=4, stride=4,
                              shuffle=False)
    inputs, targets = next(iter(dl))
    assert inputs.shape == (2, 4)
    assert targets.shape == (2, 4)
    assert inputs.dtype == torch.int64  # nn.Embedding needs integer ids
