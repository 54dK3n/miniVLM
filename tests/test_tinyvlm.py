import torch

from tiny_gpt import TinyGPT
from tinyvlm import TinyVLM
from vision_encoder import TinyViT


def test_tinyvlm_bridge_shape_loss_and_backward():
    vocab_size = 32
    vit = TinyViT(16, 8, 3, 16, 4, 1)
    gpt = TinyGPT(16, 4, 1, vocab_size, max_seq_len=9)
    model = TinyVLM(vit, gpt, vit_dim=16, gpt_dim=16)

    images = torch.randn(2, 3, 16, 16)
    input_ids = torch.randint(0, vocab_size, (2, 4))
    labels = torch.randint(0, vocab_size, (2, 4))
    labels[1, -1] = -100

    logits, loss = model(images, input_ids, labels)

    # 4 patch tokens + 1 CLS token + 4 text tokens.
    assert logits.shape == (2, 9, vocab_size)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.visual_proj.weight.grad is not None
