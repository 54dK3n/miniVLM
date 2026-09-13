import torch
import pytest

from tiny_gpt.attention import causal_mask, scaled_dot_product_attention
from tiny_gpt.multi_head_attention import MultiHeadAttention


def test_attention_shape_and_causal_mask():
    attention = MultiHeadAttention(16, 4)
    mask = causal_mask(5)
    output = attention(torch.randn(2, 5, 16), mask)
    assert output.shape == (2, 5, 16)
    assert not mask[0, 0, 0, 1]


def test_causal_mask_has_expected_lower_triangle():
    mask = causal_mask(3)
    expected = torch.tensor(
        [[True, False, False], [True, True, False], [True, True, True]]
    )
    assert mask.shape == (1, 1, 3, 3)
    assert torch.equal(mask[0, 0], expected)


def test_scaled_attention_cannot_read_future_values():
    # Zero queries and keys make the allowed attention weights uniform. The
    # first query must still see only the first value because of the mask.
    q = torch.zeros(1, 1, 2, 1)
    k = torch.zeros(1, 1, 2, 1)
    v = torch.tensor([[[[1.0], [101.0]]]])

    output = scaled_dot_product_attention(q, k, v, causal_mask(2))

    torch.testing.assert_close(output[0, 0, 0], torch.tensor([1.0]))
    torch.testing.assert_close(output[0, 0, 1], torch.tensor([51.0]))


def test_multi_head_attention_rejects_invalid_head_count():
    with pytest.raises(AssertionError):
        MultiHeadAttention(d_model=10, num_heads=3)
