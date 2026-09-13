# TinyVLM-Train

从零实现 TinyGPT、TinyViT 和 TinyVLM，并使用 Flickr8k 进行图像描述训练。

## 目录约定

- `archive/`：个人手写练习留档，仅保留 attention、vision encoder 和早期 TinyVLM demo。
- `tiny_gpt/`：TinyGPT 正式实现。
- `vision_encoder/`：TinyViT 正式实现。
- `tinyvlm/`：图文数据集、TinyVLM bridge、生成和训练实现。
- `scripts/`：训练与 overfit 实验入口。
- `tests/`：自动化测试。
- `data/raw/flickr8k/`：Flickr8k 图片和 captions。

## Flickr8k 训练

RTX 5060 Laptop 使用 CUDA 12.8 版 PyTorch：

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements-cuda.txt
```

```powershell
.venv\Scripts\python.exe scripts\run_tinyvlm_train.py `
  --image_root data\raw\flickr8k\Images `
  --caption_file data\raw\flickr8k\captions.txt `
  --batch_size 64 `
  --epochs 10 `
  --lr 3e-4 `
  --max_text_len 96 `
  --num_workers 4 `
  --save_path checkpoints\flickr8k.pt
```

正式训练入口会按图片随机划分训练集和验证集，并在每个 epoch 输出正常验证 loss、
shuffle-image loss 以及验证图片生成样例：

```powershell
.venv\Scripts\python.exe scripts\run_tinyvlm_train.py `
  --image_root data\raw\flickr8k\Images `
  --caption_file data\raw\flickr8k\captions.txt `
  --batch_size 64 `
  --epochs 10 `
  --lr 3e-4 `
  --max_text_len 96 `
  --val_ratio 0.1 `
  --num_samples 3 `
  --max_new_tokens 64 `
  --save_path checkpoints\best_tinyvlm.pt
```

## 单样本 overfit 检查

```powershell
.venv\Scripts\python.exe scripts\run_tinyvlm_overfit.py --steps 500
```

## 90/10 图片级验证训练

```powershell
.venv\Scripts\python.exe scripts\run_tinyvlm_validation.py `
  --image_root data\raw\flickr8k\Images `
  --caption_file data\raw\flickr8k\captions.txt `
  --batch_size 64 `
  --epochs 10 `
  --max_text_len 96
```

默认使用字符级 tokenizer。若要训练 word-level 模型，加入：

```powershell
  --tokenizer word --min_word_frequency 2 --max_vocab_size 10000 `
  --tie_word_embeddings
```

word-level 词表只从 training split 构建，并随 checkpoint 保存；评估脚本会自动恢复。
字符级与词级 checkpoint 的 embedding / LM head 维度不同，不能互换，切换后必须重新训练。
`--tie_word_embeddings` 让输入 embedding 与 LM 输出权重共享，可显著降低大词表模型参数量。

划分按图片进行，避免同一图片的不同 caption 泄漏到训练集和验证集。最终输出
validation loss、perplexity、token accuracy、BLEU-1、BLEU-4 及生成样例。

## 图片消融评估

```powershell
.venv\Scripts\python.exe scripts\run_image_ablation.py `
  --checkpoint checkpoints\best_tinyvlm_shuffle.pt `
  --image_root data\raw\flickr8k\Images `
  --caption_file data\raw\flickr8k\captions.txt
```

输出正常图片、batch 内乱序图片、全零图片和 `[0,1]` 均匀噪声图片四种
validation loss。可用 `--report_path logs/flickr8k_ablation.json` 保存报告。

## Multimodal KV Cache 与生成基准

TinyVLM generation 默认使用 visual-prefix KV Cache；可通过
`use_kv_cache=False` 回退到完整重算路径。

```powershell
.venv\Scripts\python.exe scripts\benchmark_vlm_generate.py `
  --checkpoint checkpoints\best_tinyvlm_shuffle.pt `
  --image_root data\raw\flickr8k\Images `
  --caption_file data\raw\flickr8k\captions.txt `
  --batch_sizes 1 8 --max_new_tokens 64 --repeats 5
```

## Toy visual grounding

```powershell
.venv\Scripts\python.exe scripts\create_toy_grounding_dataset.py

.venv\Scripts\python.exe scripts\run_tinyvlm_train.py `
  --image_root data\toy\visual_grounding\images `
  --caption_file data\toy\visual_grounding\captions.csv `
  --epochs 30 --batch_size 64 --lr 1e-3 `
  --max_text_len 32 --num_samples 1 `
  --save_path checkpoints\toy_visual_grounding.pt
```
