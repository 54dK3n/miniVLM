# TinyVLM-Train 必须手写与理解优先级

> 本文档只围绕当前项目的 11 条主线。  
> “必须手写”指不依赖 IDE，能写出核心 tensor 变换、mask 和 loss；不要求背 API 参数顺序。  
> 当前状态：1–9 已实现并有测试，10–11 是下一阶段必须掌握但尚未接入项目的内容。

## 总优先级

| 顺序 | 主题 | 要求 | 当前状态 |
|---:|---|---|---|
| 1 | Attention / MHA / causal mask | 🔴 必须手写 | 已实现 |
| 2 | TransformerBlock | 🔴 必须手写 | 已实现 |
| 3 | TinyGPT forward / loss / generate | 🔴 必须手写 | 已实现 |
| 4 | KV Cache | 🔴 必须手写核心流程 | GPT、VLM 均已实现 |
| 5 | ViT patch embedding / QKV / block | 🔴 必须手写 | 已实现 |
| 6 | TinyVLM bridge | 🔴 必须手写 | 已实现 |
| 7 | Image/text token 拼接与 Prefix Mask | 🔴 必须手写 | 已实现 |
| 8 | Text-only loss masking | 🔴 必须手写 | 已实现 |
| 9 | VLM generate | 🔴 必须手写 | 已实现，含 KV Cache |
| 10 | SFT answer-only labels | 🔴 必须手写 label 构造 | 尚未实现 |
| 11 | DPO logprob + loss | 🔴 必须手写公式与核心函数 | 尚未实现 |

---

## 1. Attention / MHA / Causal Mask

### 必须手写

Scaled Dot-Product Attention：

```text
Attention(Q,K,V) = softmax(QKᵀ / √d_head)V
```

Shape：

```text
x:       [B,T,D]
q/k/v:   [B,T,D]
拆头:     [B,H,T,Hd]
scores:  [B,H,T,T]
weights: [B,H,T,T]
output:  [B,H,T,Hd] -> [B,T,D]
Hd = D / H
```

核心代码：

```python
assert d_model % num_heads == 0
head_dim = d_model // num_heads

q = q_proj(x).view(B, T, H, head_dim).transpose(1, 2)
k = k_proj(x).view(B, T, H, head_dim).transpose(1, 2)
v = v_proj(x).view(B, T, H, head_dim).transpose(1, 2)

scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
scores = scores.masked_fill(mask == 0, float("-inf"))
weights = F.softmax(scores, dim=-1)
output = weights @ v
output = output.transpose(1, 2).contiguous().view(B, T, D)
output = out_proj(output)
```

Causal mask：

```python
mask = torch.tril(torch.ones(T, T)).bool().view(1, 1, T, T)
```

### 必须解释

- 除以 `sqrt(head_dim)`：防止点积方差随维度增大，避免 softmax 饱和。
- `softmax(dim=-1)`：每个 query 对所有 key 的权重归一化。
- mask 必须在 softmax 前应用，被禁止位置填 `-inf`。
- `transpose` 后通常不连续，合并头前使用 `contiguous()`。
- Causal mask 禁止文本位置读取未来 token。

---

## 2. TransformerBlock

### 必须手写

当前项目采用 Pre-Norm：

```python
class TransformerBlock(nn.Module):
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x
```

MLP：

```python
Linear(D, 4D) -> GELU -> Linear(4D, D)
```

### 必须解释

- 两条 residual 路径 shape 始终是 `[B,T,D]`。
- Attention 负责 token 间通信。
- MLP 只变换最后一维，负责单个 token 内部特征变换。
- Pre-Norm 的顺序是 `LN -> sublayer -> residual add`。
- ViT Block 与 GPT Block 结构相似，主要区别在 attention mask。

---

## 3. TinyGPT Forward / Loss / Generate

### 必须手写 Forward

```text
input_ids [B,T]
  -> token embedding [B,T,D]
  + position embedding [T,D]
  -> TransformerBlock × L
  -> final LayerNorm
  -> LM head
  -> logits [B,T,V]
```

```python
position_ids = torch.arange(past_len, past_len + T, device=input_ids.device)
x = token_embedding(input_ids) + position_embedding(position_ids)
for block in blocks:
    x = block(x, causal_mask)
logits = lm(final_ln(x))
```

### 必须手写 Loss

当前模型不在 forward 内部再次 shift；Dataset 必须提前构造右移 labels：

```text
input:  [x0,x1,x2,x3]
label:  [x1,x2,x3,x4]
```

```python
loss = F.cross_entropy(
    logits.reshape(-1, vocab_size),
    labels.reshape(-1),
    ignore_index=-100,
)
```

### 必须手写 Generate

```python
for _ in range(max_new_tokens):
    logits, ... = model(current_input, ...)
    next_logits = logits[:, -1, :] / temperature
    next_token = sample_or_argmax(next_logits)
    ids = torch.cat([ids, next_token], dim=1)
```

必须包含：`model.eval()`、`torch.no_grad()`、EOS 停止和最大长度保护。

---

## 4. KV Cache

### 必须手写核心流程

每层 cache：

```text
K/V: [B,H,T_cache,Hd]
```

Attention 内部：

```python
if past_kv is not None:
    past_k, past_v = past_kv
    k = torch.cat([past_k, k], dim=2)
    v = torch.cat([past_v, v], dim=2)
present_kv = (k, v)
```

TinyGPT：

```text
Prefill: 完整 prompt -> cache
Decode:  只输入 next_token + past_kvs
```

TinyVLM：

```text
Prefill: image tokens + BOS/prompt
cache:   [B,H,N_img+T_prompt,Hd]
Decode:  images=None，只输入 next_token
```

Decode mask：

```text
[1,1,T_new,T_cache+T_new]
```

### 必须解释

- Cache 保存每层已经计算的 K/V，不缓存 Q。
- 新 token 的 position id 从 `past_length` 开始。
- VLM cache 包含 visual prefix，因此 decode 不再重复运行 ViT。
- Cache 不改变模型数学结果，只减少重复计算。
- 必须能验证 cache/no-cache logits 与 greedy token 完全一致。

---

## 5. ViT Patch Embedding / QKV / Block

### 必须手写 Patch Embedding

```text
image: [B,3,H,W]
Conv2d(kernel=P, stride=P)
     -> [B,D,H/P,W/P]
flatten(2)
     -> [B,D,N]
transpose(1,2)
     -> [B,N,D]
N = (H/P)(W/P)
```

```python
self.proj = nn.Conv2d(3, D, kernel_size=P, stride=P)
x = self.proj(image).flatten(2).transpose(1, 2)
```

### 必须手写 ViT QKV

```python
qkv = nn.Linear(D, 3 * D)(x)
qkv = qkv.view(B, N, 3, H, Hd).permute(2, 0, 3, 1, 4)
q, k, v = qkv[0], qkv[1], qkv[2]
```

### 必须手写完整入口

```python
patches = patch_embed(images)
cls = cls_token.expand(B, -1, -1)
x = torch.cat([cls, patches], dim=1)
x = x + pos_embed[:, :N + 1]
for block in blocks:
    x = block(x)
return final_ln(x)  # [B,N+1,D]
```

ViT self-attention 不使用 causal mask，所有 patch 双向可见。

---

## 6. TinyVLM Bridge

### 必须手写

```python
image_tokens = vit(images)                    # [B,N_img,D_vit]
image_embeddings = visual_proj(image_tokens) # [B,N_img,D_gpt]
text_embeddings = gpt.token_embedding(ids)   # [B,T,D_gpt]
hidden = torch.cat([image_embeddings, text_embeddings], dim=1)
```

后续复用 GPT blocks、final norm 和 LM head：

```python
hidden = hidden + position_embedding(position_ids)
for block in gpt.blocks:
    hidden = block(hidden, prefix_mask)
logits = lm_or_lm_head(gpt.final_ln(hidden))
```

### 必须解释

- `visual_proj` 解决 `D_vit != D_gpt`。
- VLM 输出 shape 为 `[B,N_img+T,V]`。
- LM head 兼容项目里的 `lm` / `lm_head` 命名。
- Bridge 不改变 TinyGPT 或 TinyViT 的核心结构。

---

## 7. Image/Text Token 拼接与 Prefix Mask

### 必须手写拼接

```text
[visual token 0 ... visual token N-1, BOS, text token ...]
```

```python
x = torch.cat([image_embeddings, text_embeddings], dim=1)
```

### 必须手写 Prefix Mask

```text
                  Image keys    Text keys
Image queries     双向可见       不可见
Text queries      全部可见       causal
```

```python
mask = torch.zeros(L, L, dtype=torch.bool)
mask[:N_img, :N_img] = True
mask[N_img:, :N_img] = True
mask[N_img:, N_img:] = torch.tril(torch.ones(T, T)).bool()
mask = mask.view(1, 1, L, L)
```

必须能画出 `N_img=2,T=3` 的 5×5 mask，并解释为什么 image queries 不能读取文本。

---

## 8. Text-only Loss Masking

### 必须手写

Dataset 只返回文本 labels：

```text
input_ids = [BOS] + caption_ids
labels    = caption_ids + [EOS]
```

Collate：

```python
input_ids padding = pad_token_id
labels padding = -100
```

TinyVLM forward 内补 image labels：

```python
image_labels = torch.full((B, N_img), -100)
full_labels = torch.cat([image_labels, text_labels], dim=1)
```

### 必须解释

- `-100` 只进入 labels，绝不能进入 embedding。
- Image tokens 不直接预测词表 token，因此不参与 LM loss。
- Text padding 不参与 loss。
- Dataset 不应该提前猜测 `N_img` 并补 image labels，这是模型内部职责。

---

## 9. VLM Generate

### 必须手写

```text
输入：image
起点：[BOS]
Prefill：ViT + visual prefix + BOS -> logits + past_kvs
Decode：只输入 next_token + past_kvs
停止：EOS 或 max_new_tokens
输出：generated token ids
```

```python
logits, _, cache = model(images, bos, use_cache=True)
for _ in range(max_new_tokens):
    next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=1)
    if all_eos(next_token):
        break
    logits, _, cache = model(
        None, next_token, past_kvs=cache, use_cache=True
    )
```

### 必须解释

- Target caption 只用于展示与评估，不传入 generate。
- Greedy 是 `argmax`；sample 使用 softmax 后 multinomial。
- Cached VLM generation 中 ViT 只运行一次。
- 当前实测 KV Cache 对 32–64 token 生成约有 `1.94x–2.09x` 加速。

---

## 10. SFT Answer-only Labels

> 当前代码尚未实现；这是下一阶段要求，不能写进“已完成”。

### 必须手写

对于：

```text
prompt = "User: ... Assistant:"
answer = "a cat"
```

由于当前 TinyGPT forward 不内部 shift，数据层必须直接构造对齐后的 input/label：

```text
input_ids: [BOS, prompt tokens..., answer tokens...]
labels:    [-100 ... -100, first answer token, ..., EOS]
```

更一般地：

```python
labels = shifted_targets.clone()
labels[prompt_target_mask] = -100
labels[padding_mask] = -100
```

多模态 image prefix 的 `-100` 仍由 TinyVLM forward 内部补。

### 必须解释

- 只训练 answer，避免模型学习复读 user prompt。
- Prompt、image prefix、padding 都不能贡献 loss。
- 最容易出错的是 answer 起点与 next-token shift 相差一位。
- 必须用一个短例子逐位置写出 input_ids 和 labels 自检。

---

## 11. DPO Logprob + Loss

> 当前代码尚未实现；必须先掌握 SFT answer mask，再进入 DPO。

### 必须手写 Sequence Logprob

```python
log_probs = F.log_softmax(logits, dim=-1)
token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
sequence_logprob = (token_log_probs * answer_mask).sum(dim=-1)
```

Chosen/rejected 都只累计 answer token；prompt、image prefix、padding 不计算。

### 必须手写 DPO Loss

定义：

```text
policy_margin = logπ(chosen) - logπ(rejected)
ref_margin    = logπ_ref(chosen) - logπ_ref(rejected)
```

```text
loss = -log sigmoid(β * (policy_margin - ref_margin))
```

```python
logits = beta * (
    (policy_chosen - policy_rejected)
    - (ref_chosen - ref_rejected)
)
loss = -F.logsigmoid(logits).mean()
```

### 必须解释

- Policy 更新，reference model 冻结。
- Chosen 相对 reference 的提升超过 rejected 时，loss 下降。
- 正负号写反会让模型偏好 rejected。
- `β` 缩放 preference margin，并控制相对 reference 的约束尺度；不要只背“越大越激进”。
- DPO 不需要显式 reward model 和 PPO rollout。

---

## 最终自检顺序

每项必须回答六个问题：

1. 输入 shape 是什么？
2. 输出 shape 是什么？
3. 核心公式是什么？
4. Mask 在哪里、为什么？
5. 哪些 token 计算 loss？
6. 改一个条件时，代码哪里必须跟着变？

推荐手写顺序：

```text
Attention
-> TransformerBlock
-> TinyGPT forward/loss/generate
-> KV Cache
-> ViT
-> TinyVLM bridge
-> Prefix Mask
-> Text-only loss
-> VLM cached generate
-> SFT answer-only labels
-> DPO logprob/loss
```

不需要背：API 参数顺序、CUDA kernel、第三方大模型类名。需要真正掌握的是 shape、公式、mask、shift、loss 范围和 prefill/decode 数据流。
