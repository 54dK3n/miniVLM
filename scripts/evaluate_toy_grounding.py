import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinyvlm.dataset import ImageCaptionDataset, make_vlm_collate_fn, split_by_image
from tinyvlm.multimodal_generate import generate
from tinyvlm.toy_grounding import extract_visual_attributes, normalize_caption
from tinyvlm.train_vlm import build_model, validate
from tokenizer import tokenizer_from_checkpoint


class UniqueImageDataset(Dataset):
    def __init__(self, caption_dataset, image_names):
        self.caption_dataset = caption_dataset
        self.image_names = sorted(image_names)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        return self.caption_dataset._load_image(image_name), image_name


@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    tokenizer = tokenizer_from_checkpoint(checkpoint)
    dataset = ImageCaptionDataset(
        args.image_root,
        args.caption_file,
        tokenizer,
        image_size=config["image_size"],
        max_text_len=config["max_text_len"],
    )
    _, val_indices, _, val_images = split_by_image(
        dataset.samples,
        val_ratio=config.get("val_ratio", 0.1),
        seed=config.get("seed", 42),
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_vlm_collate_fn(tokenizer),
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = build_model(
        config["image_size"],
        config["max_text_len"],
        tokenizer.vocab_size,
        tie_word_embeddings=config.get("tie_word_embeddings", False),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float16
    val_loss, shuffle_loss, zero_loss, noise_loss = validate(
        model, val_loader, device, amp_enabled, amp_dtype
    )

    targets = {image_name: caption for image_name, caption in dataset.samples}
    image_loader = DataLoader(
        UniqueImageDataset(dataset, val_images),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    correct = {"color": 0, "shape": 0, "position": 0, "exact": 0}
    total = 0
    examples = []
    model.eval()
    for images, image_names in image_loader:
        images = images.to(device, non_blocking=True)
        generated = generate(
            model,
            images,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            max_new_tokens=config["max_text_len"] - 1,
            do_sample=False,
        )
        for ids, image_name in zip(generated.cpu(), image_names):
            prediction = tokenizer.decode(ids.tolist())
            target = targets[image_name]
            predicted_attributes = extract_visual_attributes(prediction)
            target_attributes = extract_visual_attributes(target)
            for attribute_index, attribute_name in enumerate(
                ("color", "shape", "position")
            ):
                correct[attribute_name] += int(
                    predicted_attributes[attribute_index]
                    == target_attributes[attribute_index]
                )
            correct["exact"] += int(
                normalize_caption(prediction) == normalize_caption(target)
            )
            total += 1
            if len(examples) < 8:
                examples.append(
                    {"image": image_name, "target": target, "prediction": prediction}
                )

    report = {
        "validation_images": total,
        "val_loss": val_loss,
        "val_shuffle_loss": shuffle_loss,
        "val_zero_loss": zero_loss,
        "val_noise_loss": noise_loss,
        "shuffle_gap": shuffle_loss - val_loss,
        "zero_gap": zero_loss - val_loss,
        "noise_gap": noise_loss - val_loss,
        "noise_distribution": "uniform[0,1]",
        "color_accuracy": correct["color"] / total,
        "shape_accuracy": correct["shape"] / total,
        "position_accuracy": correct["position"] / total,
        "exact_match": correct["exact"] / total,
        "examples": examples,
    }
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for key, value in report.items():
        if key != "examples":
            print(f"{key}: {value}")
    for example in examples:
        print(f"target:    {example['target']}")
        print(f"generated: {example['prediction']}")
    print(f"report: {report_path}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate toy visual grounding")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--caption_file", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--report_path", default="logs/toy_grounding_metrics.json")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
