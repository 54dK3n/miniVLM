import pytest
import torch

from tiny_gpt import TinyGPT


def test_tiny_gpt_causal_logits_ignore_future_tokens():
    torch.manual_seed(0)
    model = TinyGPT(16, 4, 2, 32, 8).eval()
    tokens_a = torch.tensor([[1, 2, 3, 4]])
    tokens_b = torch.tensor([[1, 2, 3, 9]])

    logits_a, _ = model(tokens_a)
    logits_b, _ = model(tokens_b)

    torch.testing.assert_close(logits_a[:, :3], logits_b[:, :3])


def test_cached_logits_match_full_forward():
    torch.manual_seed(1)
    model = TinyGPT(16, 4, 2, 32, 8).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])

    full_logits, _ = model(tokens)
    _, _, cache = model(tokens[:, :-1], use_cache=True)
    cached_logits, _, next_cache = model(
        tokens[:, -1:], past_kvs=cache, use_cache=True
    )

    torch.testing.assert_close(
        cached_logits[:, 0], full_logits[:, -1], rtol=1e-5, atol=1e-6
    )
    assert len(next_cache) == 2
    assert all(k.shape == (1, 4, 4, 4) for k, _ in next_cache)
    assert all(v.shape == (1, 4, 4, 4) for _, v in next_cache)


def test_tiny_gpt_rejects_sequence_longer_than_position_table():
    model = TinyGPT(16, 4, 1, 32, max_seq_len=4)

    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model(torch.ones(1, 5, dtype=torch.long))


def test_tiny_gpt_loss_backpropagates_to_embeddings():
    torch.manual_seed(2)
    model = TinyGPT(16, 4, 1, 32, 8)
    inputs = torch.randint(0, 32, (2, 4))
    targets = torch.randint(0, 32, (2, 4))

    logits, loss = model(inputs, targets)
    loss.backward()

    assert logits.shape == (2, 4, 32)
    assert loss.ndim == 0
    assert model.token_embedding.weight.grad is not None
    assert model.position_embedding.weight.grad is not None


def test_tied_embedding_is_shared_with_lm_head():
    vocab_size = 32
    d_model = 16
    untied = TinyGPT(d_model, 4, 1, vocab_size, 8)
    tied = TinyGPT(
        d_model,
        4,
        1,
        vocab_size,
        8,
        tie_word_embeddings=True,
    )

    assert tied.lm.weight is tied.token_embedding.weight
    assert sum(parameter.numel() for parameter in untied.parameters()) - sum(
        parameter.numel() for parameter in tied.parameters()
    ) == vocab_size * d_model

    inputs = torch.randint(0, vocab_size, (2, 4))
    targets = torch.randint(0, vocab_size, (2, 4))
    _, loss = tied(inputs, targets)
    loss.backward()
    assert tied.lm.weight.grad is tied.token_embedding.weight.grad
