import torch

from .patch_embedding import PatchEmbedding


def test_patch_embedding_shape():
    module = PatchEmbedding(32, 8, 3, 64)
    output = module(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 16, 64)
