import torch

from tiny_gpt import TinyGPT


def test_tiny_gpt_training_step():
    model = TinyGPT(16, 4, 1, 20, 8)
    inputs = torch.randint(0, 20, (2, 4))
    targets = torch.randint(0, 20, (2, 4))
    _, loss = model(inputs, targets)
    loss.backward()
    assert model.token_embedding.weight.grad is not None
