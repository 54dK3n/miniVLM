import torch


def get_batch(data, batch_size, block_size, device="cpu"):
    indices = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    inputs = torch.stack([data[i : i + block_size] for i in indices]).to(device)
    targets = torch.stack([data[i + 1 : i + block_size + 1] for i in indices]).to(device)
    return inputs, targets


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size=32, eval_iters=20, device="cpu"):
    model.eval()
    output = {}
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters)
        for index in range(eval_iters):
            inputs, targets = get_batch(data, batch_size, block_size, device)
            _, loss = model(inputs, targets)
            losses[index] = loss.item()
        output[split] = losses.mean().item()
    model.train()
    return output


def training_loop(model, train_data, val_data, epochs, batch_size=32, block_size=8, lr=1e-3, eval_interval=100, device="cpu"):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        inputs, targets = get_batch(train_data, batch_size, block_size, device)
        _, loss = model(inputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, block_size, batch_size, device=device)
            history.append({"epoch": epoch, "train_loss": losses["train"], "val_loss": losses["val"]})
    return history
