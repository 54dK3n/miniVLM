import torch

from .tiny_vit import TinyViT


def test_tiny_vit_shape_and_backward():
    model = TinyViT(
        img_size=32,
        patch_size=8,
        in_channels=3,
        d_model=64,
        num_heads=8,
        num_layers=2,
    )
    output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 17, 64)
    output[0, 0, 0].backward()
    assert model.patch_embed.projection.weight.grad is not None
    assert model.cls_token.grad is not None
    assert model.pos_embed.grad is not None
