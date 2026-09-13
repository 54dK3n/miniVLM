import torch

from tiny_gpt import TinyGPT
from vision_encoder import TinyViT


def test_core_output_shapes():
    language_model = TinyGPT(16, 4, 2, 20, 16)
    logits, _ = language_model(torch.randint(0, 20, (2, 4)))
    assert logits.shape == (2, 4, 20)

    vision_model = TinyViT(32, 8, 3, 64, 8, 2)
    image_tokens = vision_model(torch.randn(2, 3, 32, 32))
    assert image_tokens.shape == (2, 17, 64)
