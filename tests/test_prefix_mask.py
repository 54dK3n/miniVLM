import torch

from tinyvlm import build_decode_mask, build_prefix_mask


def test_visual_prefix_text_causal_mask():
    mask = build_prefix_mask(image_length=2, text_length=3)[0, 0]
    expected = torch.tensor(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )

    assert mask.shape == (5, 5)
    assert torch.equal(mask, expected)


def test_prefix_mask_keeps_text_future_hidden():
    mask = build_prefix_mask(image_length=3, text_length=4)[0, 0]

    # The first text query sees all visual keys and itself, but no future text.
    assert mask[3, :4].all()
    assert not mask[3, 4:].any()
    # The last text query can see the complete visual/text prefix.
    assert mask[-1].all()


def test_cached_decode_mask_sees_cache_and_not_new_future():
    mask = build_decode_mask(past_length=4, text_length=3)[0, 0]
    expected = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask, expected)
