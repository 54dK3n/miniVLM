"""从 SFT json 构造 DPO 偏好对（第一版 rejected = shuffled caption）。

读取 SFT 数据（image + prompt + answer/caption），为每条造：
    chosen   = 原始 caption
    rejected = 随机抽另一条 caption（要求 != chosen）—— 描述了图里没有的内容，作幻觉负样本
输出 data/processed/dpo_train.json，格式：
    [{"image","prompt","chosen","rejected"}, ...]

后续可把 rejected 换成 SFT 模型生成的 bad caption（hard negative），接口不变。
"""

import argparse
import json
import random
from pathlib import Path

REJECTED_STRATEGY = "shuffled caption (random other sample, != chosen)"


def extract_answer(row):
    """SFT json 里 answer 字段可能叫 caption / answer / chosen。"""
    for key in ("caption", "answer", "chosen"):
        if row.get(key):
            return row[key]
    return ""


def parse_args():
    p = argparse.ArgumentParser(description="Build DPO preference pairs from SFT json")
    p.add_argument("--input", default="data/processed/sft_train.json")
    p.add_argument("--output", default="data/processed/dpo_train.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_rows", type=int, default=None, help="最多处理多少条输入")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    if args.max_rows is not None:
        data = data[: args.max_rows]

    rng = random.Random(args.seed)
    caption_pool = [extract_answer(r).strip() for r in data]
    caption_pool = [c for c in caption_pool if c]

    pairs, skipped = [], 0
    for row in data:
        image = (row.get("image") or "").strip()
        prompt = (row.get("prompt") or "").strip()
        chosen = extract_answer(row).strip()

        # rejected：随机抽一条 != chosen 的 caption
        rejected = ""
        if caption_pool:
            for _ in range(20):
                cand = rng.choice(caption_pool)
                if cand and cand != chosen:
                    rejected = cand
                    break

        # 基本检查
        if not (image and prompt and chosen and rejected) or chosen == rejected:
            skipped += 1
            continue

        pairs.append({"image": image, "prompt": prompt, "chosen": chosen, "rejected": rejected})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"input rows:     {len(data)}")
    print(f"output pairs:   {len(pairs)}")
    print(f"skipped:        {skipped}")
    print(f"rejected by:    {REJECTED_STRATEGY}")
    print(f"written to:     {out_path}")
    if pairs:
        print("sample:")
        print(json.dumps(pairs[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
