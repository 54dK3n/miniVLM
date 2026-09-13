"""多模态 SFT 训练循环。

完全复用已有的 ``tinyvlm.TinyVLM`` —— 不重新实现模型。
与 caption 预训练(``tinyvlm/train_vlm.py``)的区别只有两点：
  1. 数据来自 ``FlickerDataset``：instruction prompt + response-only label mask，
     prompt 段已是 -100，只对 answer 算 loss；
  2. 通常从预训练好的 TinyVLM checkpoint 继续训练(可选)。

TinyVLM.forward(images, input_ids, labels) 内部已经：
  - 把 N_img 个 visual prefix 的 label 置 -100（视觉段不算 loss）；
  - 用 ignore_index=-100 做 cross_entropy（prompt + padding + 视觉段都被跳过）。
所以这里只负责喂数据、反传、更新。
"""

import torch
from torch.optim import AdamW


@torch.no_grad()
def evaluate_sft(model, dataloader, device="cpu", image_transform=None):
    """计算数据集上的 per-token 平均 response-only loss。

    传入 image_transform 可做图像消融（图像打乱 / 置零），用来验证模型是否真的
    依赖图像作答：若 normal_loss << shuffled_loss，说明视觉信息确实被用上了。
    """
    model.to(device)
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in dataloader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        if image_transform is not None:
            images = image_transform(images)
        _, loss = model(images, input_ids, labels)
        num_tokens = int((labels != -100).sum())
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
    if was_training:
        model.train()
    return total_loss / max(total_tokens, 1)


def load_pretrained(model, checkpoint_path, device="cpu", strict=True):
    """从 caption 预训练 checkpoint 加载权重，作为 SFT 起点。"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=strict)
    return model


def train_sft(
    model,
    dataloader,
    *,
    epochs=1,
    lr=1e-4,
    weight_decay=0.0,
    grad_clip=1.0,
    device="cpu",
    log_interval=10,
):
    """在 SFTCollator 产出的 batch 上微调 TinyVLM，返回每个 step 的 loss 历史。"""
    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(images, input_ids, labels)  # 复用 TinyVLM 的 response-only loss
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            history.append(loss.item())
            if step % log_interval == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
    return history
