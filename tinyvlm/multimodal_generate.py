import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model,
    images,
    bos_token_id,
    eos_token_id,
    max_new_tokens=32,
    temperature=1.0,
    do_sample=False,
    use_kv_cache=True,
):
    """Generate captions from a visual prefix, optionally using KV Cache."""
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")

    model.eval()
    batch_size = images.size(0)
    input_ids = torch.full(
        (batch_size, 1),
        bos_token_id,
        dtype=torch.long,
        device=images.device,
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=images.device)

    if use_kv_cache:
        logits, _, past_kvs = model(
            images,
            input_ids,
            labels=None,
            use_cache=True,
        )

    for step in range(max_new_tokens):
        if not use_kv_cache:
            logits, _ = model(images, input_ids, labels=None)
        next_logits = logits[:, -1, :] / temperature  # [B,V]

        if do_sample:
            probabilities = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
        else:
            next_token = next_logits.argmax(dim=-1, keepdim=True)

        # Finished batch items stay on EOS while other captions continue.
        next_token = torch.where(
            finished.unsqueeze(1),
            torch.full_like(next_token, eos_token_id),
            next_token,
        )
        input_ids = torch.cat([input_ids, next_token], dim=1)  # [B,T+1]
        finished |= next_token.squeeze(1).eq(eos_token_id)
        if finished.all():
            break

        if use_kv_cache and step < max_new_tokens - 1:
            logits, _, past_kvs = model(
                None,
                next_token,
                labels=None,
                past_kvs=past_kvs,
                use_cache=True,
            )

    return input_ids
