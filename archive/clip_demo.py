"""Handwritten TinyVLM bridge and single-pair overfit experiment."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tiny_gpt import TinyGPT
from vision_encoder import TinyViT


class TinyVLM(nn.Module):
    def __init__(self, vit, gpt, vit_dim, gpt_dim):
        super().__init__()
        self.vit = vit
        self.gpt = gpt
        self.visual_proj = nn.Linear(vit_dim, gpt_dim)

    def forward(self, img, input_ids, labels=None):
        batch_size, _ = input_ids.shape
        image_tokens = self.vit(img)  # [B,N_img,vit_dim]
        image_embeddings = self.visual_proj(image_tokens)  # [B,N_img,gpt_dim]
        image_length = image_embeddings.size(1)

        text_embeddings = self.gpt.token_embedding(input_ids)  # [B,T_text,gpt_dim]
        hidden_states = torch.cat([image_embeddings, text_embeddings], dim=1)
        total_length = hidden_states.size(1)

        position_ids = torch.arange(total_length, device=hidden_states.device)
        hidden_states = hidden_states + self.gpt.position_embedding(position_ids)
        mask = torch.tril(
            torch.ones(total_length, total_length, device=hidden_states.device)
        ).bool().view(1, 1, total_length, total_length)

        for block in self.gpt.blocks:
            hidden_states = block(hidden_states, mask)
        hidden_states = self.gpt.final_ln(hidden_states)
        logits = self.gpt.lm(hidden_states)

        loss = None
        if labels is not None:
            image_labels = torch.full(
                (batch_size, image_length),
                -100,
                device=labels.device,
                dtype=labels.dtype,
            )
            full_labels = torch.cat([image_labels, labels], dim=1)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                full_labels.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss


@torch.no_grad()
def generate(model, image, bos_token_id, eos_token_id=None, max_new_tokens=20):
    model.eval()
    input_ids = torch.full(
        (image.size(0), 1),
        bos_token_id,
        dtype=torch.long,
        device=image.device,
    )
    for _ in range(max_new_tokens):
        logits, _ = model(image, input_ids)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if eos_token_id is not None and next_token.eq(eos_token_id).all():
            break
    return input_ids


def run_overfit_demo(steps=500):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit = TinyViT(64, 8, 3, 64, 8, 2)
    gpt = TinyGPT(64, 4, 2, vocab_size=100, max_seq_len=128)
    model = TinyVLM(vit, gpt, 64, 64).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    image = torch.randn(1, 3, 64, 64, device=device)
    input_ids = torch.tensor([[1, 12, 34, 56]], device=device)
    labels = torch.tensor([[12, 34, 56, 2]], device=device)

    for step in range(steps):
        optimizer.zero_grad()
        _, loss = model(image, input_ids, labels)
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == steps - 1:
            print(step, loss.item())
    return model, image


if __name__ == "__main__":
    trained_model, demo_image = run_overfit_demo()
    print(generate(trained_model, demo_image, 1, 2, 10))
