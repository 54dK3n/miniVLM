import torch
import torch.nn.functional as F


def _sample_next_token(logits, temperature, top_k):
    logits = logits[:, -1, :] / temperature
    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, top_k)
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(1, indices, values)
        logits = filtered
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


@torch.no_grad()
def generate(
    model,
    idx,
    max_new_tokens,
    block_size,
    temperature=1.0,
    top_k=None,
    use_kv_cache=True,
):
    model.eval()
    idx_cond = idx[:, -block_size:]
    can_use_cache = (
        use_kv_cache
        and idx_cond.size(1) + max_new_tokens <= model.max_seq_len
    )

    if can_use_cache:
        logits, _, past_kvs = model(idx_cond, use_cache=True)
        for step in range(max_new_tokens):
            next_token = _sample_next_token(logits, temperature, top_k)
            idx = torch.cat((idx, next_token), dim=1)
            if step < max_new_tokens - 1:
                logits, _, past_kvs = model(
                    next_token, past_kvs=past_kvs, use_cache=True
                )
        return idx

    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -block_size:])
        next_token = _sample_next_token(logits, temperature, top_k)
        idx = torch.cat((idx, next_token), dim=1)
    return idx
