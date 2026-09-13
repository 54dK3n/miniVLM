"""Small single-pair overfit check for the TinyVLM bridge."""

import torch
from torch.optim import AdamW

from tiny_gpt import TinyGPT
from vision_encoder import TinyViT

from .multimodal_generate import generate
from .tinyvlm_model import TinyVLM


def run_overfit_demo(steps=500, lr=1e-3, device=None):
    """Overfit one synthetic image-caption pair as a gradient-flow check."""
    torch.manual_seed(42)
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    vocab_size = 100

    vit = TinyViT(64, 8, 3, 64, 8, 2)
    gpt = TinyGPT(64, 4, 2, vocab_size, max_seq_len=128)
    model = TinyVLM(vit, gpt, vit_dim=64, gpt_dim=64).to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)

    image = torch.randn(1, 3, 64, 64, device=device)
    input_ids = torch.tensor([[1, 12, 34, 56]], device=device)
    labels = torch.tensor([[12, 34, 56, 2]], device=device)

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(image, input_ids, labels)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == steps - 1:
            print(f"step={step:3d} loss={loss.item():.6f}")

    generated = generate(
        model,
        image,
        bos_token_id=1,
        eos_token_id=2,
        max_new_tokens=10,
    )
    print("generated:", generated)
    return model, image, generated
