import torch

from .generate import generate
from .tiny_gpt import TinyGPT


def test_tiny_gpt_shape_and_cache():
    model = TinyGPT(16, 4, 2, 20, 16)
    tokens = torch.randint(0, 20, (2, 3))
    logits, _, cache = model(tokens, use_cache=True)
    assert logits.shape == (2, 3, 20)
    assert len(cache) == 2
    next_logits, _, next_cache = model(
        torch.randint(0, 20, (2, 1)), past_kvs=cache, use_cache=True
    )
    assert next_logits.shape == (2, 1, 20)
    assert all(layer[0].shape[2] == 4 for layer in next_cache)


def test_cached_generation_matches_no_cache():
    torch.manual_seed(7)
    model = TinyGPT(16, 4, 2, 20, 16).eval()
    prompt = torch.randint(0, 20, (1, 4))
    torch.manual_seed(8)
    cached = generate(model, prompt, 5, 16, top_k=5, use_kv_cache=True)
    torch.manual_seed(8)
    uncached = generate(model, prompt, 5, 16, top_k=5, use_kv_cache=False)
    assert torch.equal(cached, uncached)
