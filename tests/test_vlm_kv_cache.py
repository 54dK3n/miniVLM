import torch

from tiny_gpt import TinyGPT
from tinyvlm import TinyVLM, generate
from vision_encoder import TinyViT


def _build_test_model():
    vit = TinyViT(16, 8, 3, 16, 4, 1)
    gpt = TinyGPT(16, 4, 1, 32, max_seq_len=13)
    return TinyVLM(vit, gpt, vit_dim=16, gpt_dim=16).eval()


def test_cached_decode_logits_match_full_multimodal_forward():
    torch.manual_seed(4)
    model = _build_test_model()
    images = torch.randn(2, 3, 16, 16)
    input_ids = torch.randint(0, 32, (2, 4))

    full_logits, _ = model(images, input_ids)
    _, _, cache = model(images, input_ids[:, :-1], use_cache=True)
    cached_logits, _, next_cache = model(
        None,
        input_ids[:, -1:],
        past_kvs=cache,
        use_cache=True,
    )

    torch.testing.assert_close(
        cached_logits[:, -1], full_logits[:, -1], rtol=1e-5, atol=1e-6
    )
    # 5 visual tokens + 4 text tokens are cached after decode.
    assert all(key.size(2) == 9 for key, _ in next_cache)
    assert all(value.size(2) == 9 for _, value in next_cache)


def test_cached_and_uncached_greedy_generation_match():
    torch.manual_seed(5)
    model = _build_test_model()
    images = torch.randn(2, 3, 16, 16)

    cached = generate(
        model, images, bos_token_id=1, eos_token_id=-1, max_new_tokens=5
    )
    uncached = generate(
        model,
        images,
        bos_token_id=1,
        eos_token_id=-1,
        max_new_tokens=5,
        use_kv_cache=False,
    )
    assert torch.equal(cached, uncached)


def test_cached_generation_runs_vit_only_once():
    model = _build_test_model()
    images = torch.randn(1, 3, 16, 16)
    vit_calls = 0

    def count_calls(_module, _inputs, _output):
        nonlocal vit_calls
        vit_calls += 1

    handle = model.vit.register_forward_hook(count_calls)
    generate(model, images, 1, -1, max_new_tokens=4, use_kv_cache=True)
    handle.remove()

    assert vit_calls == 1
