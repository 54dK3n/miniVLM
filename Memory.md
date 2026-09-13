# TinyVLM-Train 项目计划：从零实现多模态理解模型的预训练、后训练与代码 Review 流程

## 0. 项目定位

项目名称：**TinyVLM-Train**

目标岗位：

- 多模态理解算法工程师
- 大模型预训练算法工程师
- 大模型后训练算法工程师
- VLM / LLM 算法实习生

项目目标不是训练一个效果很强的大模型，而是完整手写并理解以下核心链路：

1. decoder-only Transformer 语言模型预训练
2. CLIP-style 图文对比预训练
3. Q-Former-lite / Cross-Attention 视觉语言桥接模块
4. 多模态 SFT 指令微调
5. 多模态 DPO 偏好优化
6. Retrieval / VQA / OCR / Counting / Hallucination 评测
7. 人工代码 Review、单元测试、错误定位和实验复盘

本项目的核心要求：**核心算法部分必须手写，并且每个模块必须经过人工 review。不能只是两个大模型互相复制粘贴。**

------

## 1. 严格规则：禁止纯复制粘贴

### 1.1 允许使用 AI 的方式

AI 可以用于：

- 解释论文和公式
- 生成代码初稿
- 提供测试样例
- 帮忙找 bug
- 帮忙设计 ablation
- 帮忙整理 README
- 模拟面试追问

### 1.2 不允许使用 AI 的方式

不允许：

- AI 写完代码后直接提交
- 不看 tensor shape 就运行
- 不懂 loss 公式就加入项目
- 不懂 mask 逻辑就训练
- 不写单元测试
- 不做小样本 overfit test
- 不记录代码 review 过程
- 直接调用 HuggingFace Trainer / PEFT / TRL 掩盖核心算法
- 调用现成 CLIP / LLaVA / Qwen-VL 当作自己的核心项目

### 1.3 每个模块提交前必须回答 6 个问题

每次完成一个模块，你必须在 REVIEW_LOG.md 里回答：

```text
1. 这个模块输入 tensor shape 是什么？
2. 输出 tensor shape 是什么？
3. 核心公式是什么？
4. mask 在哪里用？为什么这样用？
5. loss 是对哪些 token / 样本计算的？哪些不算？
6. 我写了什么测试证明它是对的？
```

如果答不上来，就不能算完成。

------

## 2. 代码 Review 工作流

每个模块必须走 5 步：

```text
Step 1：自己先写模块说明
Step 2：让 AI 生成代码初稿
Step 3：自己逐行 review tensor shape、公式、mask、loss
Step 4：写单元测试和 toy overfit test
Step 5：把 review 结果写进 REVIEW_LOG.md
```

### 2.1 AI 生成代码前，你必须先写设计说明

模板：

```md
## Module Design: [模块名]

### Goal
这个模块要解决什么问题？

### Inputs
- 输入 1: shape = ...
- 输入 2: shape = ...

### Outputs
- 输出: shape = ...

### Core Formula
写出核心公式。

### Mask Logic
是否需要 mask？mask shape 是什么？mask 中 0 / 1 或 True / False 分别代表什么？

### Edge Cases
- batch size = 1 是否能跑？
- sequence length = 1 是否能跑？
- padding 是否会影响 loss？

### Tests
计划写哪些测试？
```

------

## 3. 模块一：TinyLLM 预训练

### 3.1 目标

从零实现一个小型 decoder-only Transformer，用 next-token prediction 做文本预训练。

### 3.2 必须手写

- Token embedding
- Position embedding 或 RoPE
- Multi-head causal self-attention
- Causal mask
- MLP / SwiGLU
- LayerNorm / RMSNorm
- Residual connection
- LM head
- Next-token loss
- Training loop
- Generation loop
- KV cache inference，作为加分项

### 3.3 核心检查点

#### Attention shape 检查

```text
input:  x shape = [B, T, C]
q:      [B, n_heads, T, head_dim]
k:      [B, n_heads, T, head_dim]
v:      [B, n_heads, T, head_dim]
score:  [B, n_heads, T, T]
mask:   [1, 1, T, T]
out:    [B, T, C]
```

你必须检查：

- `C % n_heads == 0`
- attention score 是否除以 `sqrt(head_dim)`
- causal mask 是否禁止当前位置看未来 token
- softmax 是否在最后一维做
- dropout 是否只在训练时生效

#### Label shift 检查

next-token prediction 必须是：

```text
输入 tokens:  [x0, x1, x2, x3]
预测目标:     [x1, x2, x3, x4]
```

代码里通常是：

```python
logits = logits[:, :-1, :]
labels = input_ids[:, 1:]
```

你必须检查：

- logits 和 labels 的长度是否一致
- padding token 是否被 ignore
- loss 是否是对 vocabulary 维度做 cross entropy

### 3.4 必须写的测试

```text
Test 1：输入 [B, T]，输出 logits shape 必须是 [B, T, vocab_size]
Test 2：causal mask 中上三角必须被 mask
Test 3：batch size = 1 能运行
Test 4：sequence length = 1 能运行
Test 5：用 10 条重复文本 overfit，loss 必须明显下降
Test 6：generate 不报错，输出长度正确
```

------

## 4. 模块二：CLIP-style 图文对比预训练

### 4.1 目标

实现图文双塔模型，将 image embedding 和 text embedding 对齐到同一个语义空间。

### 4.2 必须手写

- Image encoder：轻量 CNN / ViT
- Text encoder：Transformer encoder
- Projection head
- L2 normalization
- Learnable temperature
- Similarity matrix
- Symmetric InfoNCE loss
- Recall@K 评测

### 4.3 核心公式

```python
image_emb = normalize(image_encoder(image))
text_emb = normalize(text_encoder(text))

logits = image_emb @ text_emb.T / temperature
labels = torch.arange(batch_size)

loss_i2t = cross_entropy(logits, labels)
loss_t2i = cross_entropy(logits.T, labels)
loss = (loss_i2t + loss_t2i) / 2
```

### 4.4 核心检查点

你必须检查：

- image embedding shape 是否是 `[B, D]`
- text embedding shape 是否是 `[B, D]`
- similarity matrix 是否是 `[B, B]`
- labels 是否是 `[0, 1, 2, ..., B-1]`
- 对角线是否代表正样本
- 非对角线是否代表 batch 内负样本
- loss 是否双向计算
- temperature 是否为正数
- embedding 是否做了 normalize

### 4.5 必须写的测试

```text
Test 1：image_emb 和 text_emb shape 都是 [B, D]
Test 2：similarity matrix shape 是 [B, B]
Test 3：labels 是 arange(B)
Test 4：如果 image_emb == text_emb，正样本相似度应该更高
Test 5：loss_i2t 和 loss_t2i 都能反向传播
Test 6：小数据集 overfit 后 Recall@1 应该上升
```

------

## 5. 模块三：Q-Former-lite / Visual Adapter

### 5.1 目标

实现视觉语言桥接模块，把 image tokens 转换成 LLM 能理解的 visual prefix。

### 5.2 推荐结构

```text
image
 -> image encoder
 -> image patch tokens
 -> learnable query tokens
 -> cross-attention
 -> visual query embeddings
 -> projection to LLM hidden size
 -> prepend to text tokens
 -> LLM generates answer
```

### 5.3 必须手写

- Learnable query tokens
- Cross-attention
- Query transformer block
- Visual projection layer
- Visual prefix injection
- Prefix attention mask

### 5.4 Cross-Attention shape 检查

```text
query tokens: [B, Q, C]
image tokens: [B, N, C]

Q = query projection(query tokens) -> [B, heads, Q, head_dim]
K = key projection(image tokens)   -> [B, heads, N, head_dim]
V = value projection(image tokens) -> [B, heads, N, head_dim]

attention score: [B, heads, Q, N]
output:          [B, Q, C]
```

你必须检查：

- cross-attention 的 Q 来自 query tokens
- K/V 来自 image tokens
- 它不是 self-attention
- output token 数量是 Q，不是 image patch 数量 N
- projection 后维度必须等于 LLM hidden size

### 5.5 必须写的测试

```text
Test 1：输入 image tokens [B, N, C]，输出 visual prefix [B, Q, llm_dim]
Test 2：Q 改变时，输出 token 数量跟着改变
Test 3：image tokens 改变时，visual prefix 也应该改变
Test 4：visual prefix 能和 text embeddings concat
Test 5：concat 后 attention mask shape 正确
```

------

## 6. 模块四：多模态 SFT

### 6.1 目标

让模型学习 image + instruction -> answer。

### 6.2 数据格式

```json
{
  "image": "xxx.jpg",
  "instruction": "What is the object on the left?",
  "answer": "A red car."
}
```

### 6.3 输入格式

```text
<image_prefix>
User: What is the object on the left?
Assistant: A red car.
```

### 6.4 核心要求：response-only loss

不能对全部 token 算 loss。必须只对 answer 部分算 loss。

```python
labels = input_ids.clone()
labels[prompt_positions] = -100
labels[padding_positions] = -100
loss = cross_entropy(logits, labels, ignore_index=-100)
```

你必须检查：

- image prefix 不算 loss
- user prompt 不算 loss
- padding 不算 loss
- answer 才算 loss
- labels 和 logits shift 是否正确

### 6.5 必须写的测试

```text
Test 1：prompt 部分 labels 必须是 -100
Test 2：answer 部分 labels 必须保留 token id
Test 3：padding 部分 labels 必须是 -100
Test 4：loss 只来自 answer tokens
Test 5：用 20 条固定 VQA 数据 overfit，模型能记住答案
```

------

## 7. 模块五：多模态 DPO

### 7.1 目标

通过 chosen / rejected 回答对减少模型幻觉，提高回答偏好质量。

### 7.2 数据格式

```json
{
  "image": "dog.jpg",
  "prompt": "Describe the image.",
  "chosen": "A dog is sitting on the grass.",
  "rejected": "A cat is sitting on the sofa."
}
```

### 7.3 必须手写

- chosen sequence log probability
- rejected sequence log probability
- reference model log probability
- DPO loss
- beta 参数
- DPO 前后对比评测

### 7.4 核心检查点

你必须检查：

- policy model 会更新
- reference model 必须冻结
- chosen / rejected 的 logprob 只对 answer 部分计算
- prompt 不算 logprob
- padding 不算 logprob
- DPO loss 的正负号不能写反
- beta 改变会影响优化强度

### 7.5 必须写的测试

```text
Test 1：reference model 参数 requires_grad=False
Test 2：chosen 和 rejected logprob shape 是 [B]
Test 3：如果 chosen logprob 明显高于 rejected，loss 应该更小
Test 4：如果 rejected logprob 高于 chosen，loss 应该更大
Test 5：DPO 训练后 chosen/rejected gap 应该扩大
```

------

## 8. REVIEW_LOG.md 模板

每个模块都写一段。

```md
# REVIEW_LOG

## Module: Causal Self-Attention

### Date
2026-xx-xx

### What AI generated
AI 生成了 attention.py 的初稿，包括 qkv projection、causal mask、softmax 和 output projection。

### My review

#### 1. Tensor shapes
- input x: [B, T, C]
- q/k/v: [B, heads, T, head_dim]
- attention score: [B, heads, T, T]
- output: [B, T, C]

#### 2. Formula check
attention = softmax(QK^T / sqrt(d)) V

#### 3. Mask check
causal mask 是上三角 mask，保证 token t 不能看到 t+1 之后的 token。

#### 4. Bugs found
- AI 初稿中 softmax dim 写成 dim=-2，应该是 dim=-1。
- mask dtype 和 attention score dtype 不一致，已修正。

#### 5. Tests added
- test_output_shape
- test_causal_mask
- test_no_future_attention

#### 6. Conclusion
该模块通过 shape test 和 mask test，可以进入训练阶段。
```

------

## 9. 让 AI 写代码时使用的 Prompt 模板

### 9.1 生成代码 Prompt

```text
你是我的代码实现助手。请只实现我指定的模块，不要额外扩展。

模块名称：[模块名]
目标：[目标]
输入 shape：[输入 shape]
输出 shape：[输出 shape]
核心公式：[公式]

限制：
1. 使用 PyTorch。
2. 不使用 HuggingFace Trainer。
3. 不调用现成大模型封装。
4. 核心算法必须手写。
5. 代码要包含 shape 注释。
6. 同时给出最小单元测试。

请输出：
1. 实现代码
2. 每一步 tensor shape
3. 单元测试
4. 可能出错的地方
```

### 9.2 代码 Review Prompt

```text
你是我的严格代码 reviewer。请检查下面代码是否符合算法公式和 tensor shape。

重点检查：
1. 输入输出 shape 是否正确
2. mask 是否正确
3. loss 是否算在正确位置
4. 是否有 broadcasting bug
5. 是否有 silent bug
6. 是否有 gradient 断掉的问题
7. 是否有训练时看起来能跑但算法错了的问题
8. 应该增加哪些单元测试

不要重写全部代码，先指出问题，再给最小修改建议。

代码如下：
[粘贴代码]
```

------

## 10. 判断自己是否真的懂代码的标准

一个模块不算完成，除非你能做到：

```text
1. 不看代码，画出模型 forward 流程
2. 写出每一步 tensor shape
3. 写出核心 loss 公式
4. 解释 mask 为什么这样写
5. 解释哪些 token 参与 loss，哪些不参与
6. 解释一个常见 bug 会导致什么现象
7. 用 toy example 证明代码是对的
8. 面试官改一个条件，你能说出代码哪里要改
```

------

## 11. 简历项目描述

```md
### TinyVLM-Train：从零实现多模态理解模型的预训练与后训练框架
**Python / PyTorch / Transformer / CLIP / Q-Former / SFT / DPO / VQA**

- 从零实现 decoder-only Transformer 语言模型，覆盖 causal self-attention、RMSNorm、SwiGLU、RoPE、KV cache、next-token prediction loss 与 perplexity evaluation，完成小规模文本预训练实验。
- 实现 CLIP-style 图文对比预训练框架，手写 image/text dual encoder、projection head、temperature scaling 与 symmetric InfoNCE loss，通过 Recall@K 和 similarity matrix 分析跨模态对齐效果。
- 设计 Q-Former-lite 视觉语言桥接模块，使用 learnable query tokens 通过 cross-attention 从图像 patch features 中抽取视觉语义，并投影到 LLM hidden space 作为 visual prefix。
- 构建 image-instruction-answer 多模态 SFT 流程，手写 prompt template、label mask 与 response-only loss，使模型学习 caption、VQA、OCR、counting 和 spatial reasoning 任务。
- 实现多模态 DPO 偏好优化，构造 chosen / rejected 图文回答对，手写 sequence log probability 与 DPO loss，对比 SFT 与 DPO 在 hallucination、OCR 和 counting 场景下的回答差异。
- 建立人工代码 Review 流程，每个核心模块均记录公式推导、tensor shape、mask 逻辑、单元测试和 toy overfit 结果，避免仅依赖大模型生成代码。
```

------

## 12. 最低完成标准

如果时间不够，至少完成：

```text
1. TinyLLM 能训练，loss 能下降
2. CLIP-style loss 正确，Recall@K 能跑
3. Q-Former-lite forward shape 正确
4. SFT label mask 正确
5. DPO loss 能在 toy data 上验证方向正确
6. 每个模块都有 REVIEW_LOG
7. README 能讲清楚算法链路
```

这就是最低可投版本。

------

## 13. 最高优先级

优先级从高到低：

```text
1. 算法正确
2. 你能解释
3. 有测试证明
4. 有实验记录
5. 代码结构清晰
6. 效果尽量好
7. UI / Demo
```

不要反过来。这个项目不是为了做一个花哨展示页，而是为了让你在预训练、多模态理解、后训练岗位面试中能撑住深挖。