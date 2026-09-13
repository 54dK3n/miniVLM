import torch


def save_checkpoint(path, model, config, stoi=None, itos=None):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "stoi": stoi,
            "itos": itos,
        },
        path,
    )
