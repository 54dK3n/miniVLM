# TinyVLM-Train Review Log

> 当前唯一复习顺序以 `TinyVLM_复习优先级.md` 的 11 项为准。  
> 本文件保留 `archive/attention_scratch.py` 以来的实现、测试、训练和性能证据。
> 分类标准：🔴 **手撕级**（必须能徒手写出来）| 🟡 **理解级**（知道原理/能讲清楚，不要求手撕）

---

## 零、Canonical Review 主线

以下顺序覆盖从单模态 Transformer 到多模态训练，再到后续 SFT/DPO 的依赖关系：

| 顺序 | Review 项 | 当前验收状态 |
|---:|---|---|
| 1 | Attention / MHA / causal mask | ✅ 实现、数值测试通过 |
| 2 | TransformerBlock | ✅ 实现、forward/backward 通过 |
| 3 | TinyGPT forward / loss / generate | ✅ 实现、训练与生成通过 |
| 4 | KV Cache | ✅ TinyGPT 与 TinyVLM 均实现 |
| 5 | ViT patch embedding / QKV / block | ✅ 实现、shape/梯度通过 |
| 6 | TinyVLM bridge | ✅ 实现、overfit/grounding 通过 |
| 7 | Image/text token 拼接 | ✅ Prefix Mask 已实现 |
| 8 | Text-only loss masking | ✅ image/padding mask 已实现 |
| 9 | VLM generate | ✅ Greedy/sample 与 KV Cache 已实现 |
| 10 | SFT answer-only labels | ⏳ 必须手写，项目尚未实现 |
| 11 | DPO logprob + loss | ⏳ 必须手写，项目尚未实现 |

Review 时不再按 CLIP、Q-Former、RoPE 等未进入当前核心实现的主题分散复习。每完成一项必须留下：shape、公式、mask、loss 范围、测试和一个典型 bug。

---

## 二、🔴 手撕级 (必须能徒手写出来)

### 1. Multi-Head Attention 完整实现

这是 Transformer 的核心，面试必问、必手撕。

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0                    # 必须整除
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads                     # 每个头的维度
        self.q_projection = nn.Linear(d_model, d_model)     # Q 投影
        self.k_projection = nn.Linear(d_model, d_model)     # K 投影
        self.v_projection = nn.Linear(d_model, d_model)     # V 投影
        self.out_projection = nn.Linear(d_model, d_model)   # 输出投影

    def forward(self, x, mask=None):
        B, T, D = x.shape                                   # (batch, seq_len, d_model)

        # 1. 线性投影
        q = self.q_projection(x)
        k = self.k_projection(x)
        v = self.v_projection(x)

        # 2. reshape → 多头: (B, T, num_heads, d_k) → (B, num_heads, T, d_k)
        q = q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        output = self.scaled_dot_attention(q, k, v, mask)

        # 4. 合并多头: (B, num_heads, T, d_k) → (B, T, d_model)
        output = output.transpose(1, 2).contiguous().view(B, T, D)

        # 5. 最终线性投影
        output = self.out_projection(output)
        return output
```

**核心要点：**
- `transpose(1,2)` 把 num_heads 维度换到 batch 后面，让每个头独立计算
- `contiguous()` 在 `view()` 之前必须调用，因为 transpose 后内存不连续
- 输出投影 `out_projection` 让多头信息融合

### 2. Scaled Dot-Product Attention（核心公式）

面试必问公式：**Attention(Q, K, V) = softmax(QK^T / √d_k) V**

```python
def scaled_dot_attention(self, q, k, v, mask=None):
    # QK^T / √d_k
    mid_output = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))

    # 因果 mask / padding mask
    if mask is not None:
        mid_output = mid_output.masked_fill(mask == 0, float('-inf'))

    # softmax 归一化
    attention_weights = F.softmax(mid_output, dim=-1)

    # 加权求和
    return attention_weights @ v
```

**核心要点：**
- `k.transpose(-2, -1)`：对最后两维做转置，实现 Q×K^T
- `math.sqrt(q.size(-1))`：除以 √d_k 防止点积值过大导致 softmax 梯度消失
- `masked_fill(mask==0, -inf)`：被 mask 的位置在 softmax 后变为 0
- `dim=-1`：在最后一个维度（每个 token 对所有 token 的注意力分布）上做 softmax

### 3. 因果掩码 (Causal Mask)

```python
def mask_padding(self, T):
    mask = torch.tril(torch.ones(T, T)).bool()    # 下三角为 True
    return mask.view(1, 1, T, T)                   # 适配 (B, heads, T, T)
```

**核心要点：**
- `torch.tril` 生成下三角矩阵，确保 token i 只能看到 ≤i 的位置
- shape: `(1, 1, T, T)` → 适配多头注意力的 4D mask 形状

### 4. Transformer Block（Pre-Norm 结构）

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)           # Attention 前的 LayerNorm
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)           # FFN 前的 LayerNorm
        self.mlp = FeedForward(d_model, d_model * ratio)

    def forward(self, x, mask=None):
        # Pre-Norm: LN → Attention → Residual
        x = self.attn(self.ln1(x), mask) + x
        # Pre-Norm: LN → FFN → Residual
        x = self.mlp(self.ln2(x)) + x
        return x
```

**核心要点：**
- **Pre-Norm** vs **Post-Norm**：GTP 系列用 Pre-Norm（LN 在子层之前），训练更稳定
- 两个残差连接，防止梯度消失
- FFN 的 hidden_dim = d_model × 4（标准做法）

### 5. FeedForward (FFN)

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),   # 升维
            nn.GELU(),                         # 激活函数
            nn.Linear(hidden_dim, d_model)     # 降维
        )
    def forward(self, x):
        return self.net(x)
```

**核心要点：**
- 两个线性层 + 激活函数
- 升维比例通常为 4×（`hidden_dim = d_model × 4`）
- GPT 系列使用 **GELU** 而非 ReLU

### 6. Token + Position Embedding

```python
# 词嵌入
token_embedding = self.token_embedding(inputs_id)    # (B, T) → (B, T, d_model)

# 位置嵌入
position_ids = torch.arange(T, device=inputs_id.device)
position_embedding = self.position_embedding(position_ids)  # (T,) → (T, d_model)

# 相加融合
x = token_embedding + position_embedding
```

**核心要点：**
- Token Embedding 和 Position Embedding 是**相加**关系，不是拼接
- Position Embedding 是可学习的（learned positional embedding），不是正弦编码
- 广播机制：`(B,T,d_model) + (T,d_model)` → `(B,T,d_model)`

### 7. 训练循环 (Training Loop)

```python
def training_loop(model, train_data, val_data, epochs, batch_size, block_size, lr, eval_interval, device):
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        x, y = get_batch(train_data, batch_size, block_size, device)
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**核心要点：**
- `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()` 三步曲
- 使用 **AdamW**（带 weight decay 的 Adam）
- `y` 是 `x` 右移一位（next token prediction）

### 8. KV Cache 是什么 / 为什么重要

代码中 benchmark 标注了 **"No KV Cache"**，面试必问。

**核心原理：**
- 自回归生成时，每生成一个新 token，要对**整个序列**重新计算 Attention
- KV Cache 缓存之前的 K 和 V，每步只计算新 token 的 Q
- 复杂度从 O(n²) 降到 O(n) per step

**关键维度变化：**
- 无 KV Cache：每步 Q,K,V 形状都是 `(B, num_heads, all_T, d_k)`
- 有 KV Cache：每步 Q 形状 `(B, num_heads, 1, d_k)`，K,V 形状 `(B, num_heads, past_T + 1, d_k)`

---

## 三、🟡 理解级 (知道原理，不要求手撕)

### 1. Tokenizer（字符级分词器）

```python
def tokenizer(text):
    chars = sorted(list(set(text)))              # 所有唯一字符
    stoi = {ch: i for i, ch in enumerate(chars)} # 字符→ID
    itos = {i: ch for ch, i in stoi.items()}     # ID→字符
    def encode(s):
        return torch.tensor([stoi[c] for c in s], dtype=torch.long)
    def decode(ids):
        return ''.join(itos[int(i)] for i in ids)
    return encode, decode, len(chars)
```

**核心要点：**
- 字符级 tokenizer：vocab_size 很小（~几十），但序列很长
- BPE (Byte Pair Encoding) vs 字符级：BPE 是 GPT-2/3 的 tokenizer，vocab_size ~50K
- 面试可能问 "为什么 GPT 用 BPE 而不是字符级" → 压缩信息密度，减少序列长度

### 2. Weight Tying / Parameter Sharing

```python
self.token_embedding = nn.Embedding(vocab_size, d_model)
self.lm = nn.Linear(d_model, vocab_size)
```

**注意：** 此代码**没有**做 weight tying。面试常考点：
- Weight tying = 让 `token_embedding.weight` 和 `lm.weight` 共享参数
- 好处：减少参数量（`vocab_size × d_model`），提升泛化
- GPT-2 做了 weight tying

### 3. register_buffer vs nn.Parameter

```python
self.register_buffer("casual_mask", mask)
```

**核心要点：**
- `nn.Parameter`：需要梯度，会被 optimizer 更新
- `register_buffer`：不需要梯度，随模型移动设备（`.to(device)`），会被 `state_dict()` 保存

### 4. 损失函数 (Cross Entropy)

```python
loss = F.cross_entropy(
    logits.view(-1, self.vocab_size),   # (B*T, vocab_size)
    targets.view(-1)                     # (B*T,)
)
```

**核心要点：**
- `view(-1, vocab_size)` 把 batch 和序列维度展平，每个 token 独立计算 loss
- Cross Entropy = LogSoftmax + NLLLoss
- 面试可能问 "为什么不自己做 softmax" → `cross_entropy` 内部做了 log-softmax，数值更稳定

### 5. Perplexity (困惑度)

```python
def compute_complexity(loss):
    return math.exp(loss)
```

**核心要点：**
- PPL = exp(交叉熵损失)
- PPL 越小越好，PPL=1 表示完美预测
- 面试可能会问 "train loss 和 val loss 差距大说明什么" → 过拟合

### 6. top-k 采样

```python
if top_k is not None:
    values, indices = torch.topk(logits, top_k)
    filtered_logits = torch.full_like(logits, float("-inf"))
    filtered_logits.scatter_(1, indices, values)
    logits = filtered_logits
probs = F.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

**核心要点：**
- 只保留概率最高的 k 个 token，其余设为 -inf
- 配合 temperature 控制生成多样性
- temperature < 1：更确定（保守），temperature > 1：更多样（冒险）

### 7. 完整模型架构 (TinyGPT)

```python
TinyGPT(
  Embedding → Position Embedding → [TransformerBlock × num_layers] → LayerNorm → Linear → logits
)
```

**输入输出流程：**
- 输入：`(B, T)` token IDs
- Token Embedding: `(B, T, d_model)`
- Position Embedding: `(T, d_model)`，广播相加
- N 个 TransformerBlock（Pre-Norm + MHA + FFN + Residual）
- Final LayerNorm → Linear(d_model, vocab_size) → `(B, T, vocab_size)`

### 8. 训练技巧清单

| 技巧 | 代码体现 | 面试要点 |
|------|---------|---------|
| Pre-Norm | `x = attn(ln(x)) + x` | 训练更稳定，GPT-2 开始使用 |
| AdamW | `torch.optim.AdamW` | 解耦 weight decay，防止 L2 正则和 Adam 耦合 |
| GELU | `nn.GELU()` | 比 ReLU 更平滑，GPT 系列标配 |
| LayerNorm | `nn.LayerNorm(d_model)` | 对最后维度归一化，不依赖 batch |
| 残差连接 | `x = sublayer(x) + x` | 缓解梯度消失，使深层网络可训练 |
| 因果掩码 | `torch.tril` | 防止未来信息泄漏 |

---

## 四、面试高频问题速查

| # | 问题 | 答案关键词 |
|---|------|----------|
| 1 | Attention 公式是什么？ | softmax(QK^T/√d_k) V |
| 2 | 为什么除以 √d_k？ | 防止点积过大 → softmax 梯度消失 |
| 3 | Multi-Head 的作用？ | 不同头关注不同子空间，捕获多种关系 |
| 4 | Pre-Norm vs Post-Norm？ | Pre-Norm 训练更稳定；Post-Norm 是原始 Transformer |
| 5 | KV Cache 原理？ | 缓存历史 K,V，每步只算新 token 的 Q；O(n²)→O(n) |
| 6 | 残差连接为什么重要？ | 缓解梯度消失，让深层模型可训练 |
| 7 | LayerNorm vs BatchNorm？ | LN 对特征维归一化，不依赖 batch；序列建模用 LN |
| 8 | GELU vs ReLU？ | GELU 更平滑，对负值有非零输出，GPT 系列标配 |
| 9 | Weight Tying 是什么？ | Embedding 和 LM head 共享权重，减少参数 |
| 10 | PPL 是什么？ | 困惑度 = exp(loss)，衡量模型对下一个 token 的不确定性 |

---

## 五、关键维度变换汇总

```
输入:                         (B, T)
Token + Pos Embedding:       (B, T, d_model)
Q/K/V 投影后:                (B, T, d_model)
view + transpose → 多头:     (B, num_heads, T, d_k)
QK^T:                        (B, num_heads, T, T)
softmax(QK^T/√d_k) V:       (B, num_heads, T, d_k)
transpose + view → 合并:     (B, T, d_model)
FFN 输出:                    (B, T, d_model)
LM head:                     (B, T, vocab_size)
```

---

*最后更新: 2026-06-21*

---

## 六、基于当前实现的面试知识三档（TinyGPT + Vision Encoder）

> 更新日期：2026-06-22  
> 依据：`Memory.md`、`tiny_gpt/`、`vision_encoder/`、`archive/vision_encoder_scratch.py`。  
> 本节是当前版本的复习主清单；前文保留作为 TinyGPT 的历史运行与 review 记录。

### 分类标准

- 🔴 **第一档：必须会手撕**。涉及核心数学变换、tensor shape、mask、残差与监督信号。不能只背代码，必须能解释每一步。
- 🟡 **第二档：必须理解，但不用背代码**。涉及训练/推理流程、工程协议和优化思想。应能解释作用、输入输出和常见 bug。
- ⚪ **第三档：可以不 care，现场查资料**。涉及 API 参数、底层 kernel、工程样板和当前项目未使用的实现细节。

> 注意：CLIP、Image Projector、Multimodal Fusion、SFT、DPO 是 `Memory.md` 中的后续核心目标。它们目前尚未实现，因此不计入“当前已掌握”，也不能因为当前没写就归到第三档。

---

## 七、🔴 第一档：必须会手撕

### 1. Scaled Dot-Product Attention

必须能写出并解释：

```text
Attention(Q,K,V) = softmax(QKᵀ / √d_head)V
```

shape 流动：

```text
x:       [B, T, D]
q/k/v:   [B, H, T, Hd]
scores:  [B, H, T, T]
output:  [B, H, T, Hd] -> [B, T, D]
Hd = D / H
```

核心手撕代码：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)

        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)
```

必须能回答：

- 为什么除以 `sqrt(head_dim)`：防止维度增大后点积方差过大、softmax 饱和。
- 为什么 `softmax(dim=-1)`：对每个 query 所能关注的 key 归一化。
- 为什么 `transpose` 后要 `contiguous()`：转置后的内存布局通常不连续，直接 `view` 不安全。
- 为什么要求 `d_model % num_heads == 0`：每个 head 必须获得相同的 `head_dim`。

### 2. Causal Mask

TinyGPT 是 decoder-only 模型，位置 `t` 不能读取未来 token。

```python
mask = torch.tril(torch.ones(T, T)).bool()
mask = mask.view(1, 1, T, T)
scores = scores.masked_fill(mask == 0, float("-inf"))
weights = F.softmax(scores, dim=-1)
```

必须理解：

- mask 在 softmax **之前**应用。
- `True/1` 表示允许关注，`False/0` 表示禁止关注。
- mask 是下三角；真正被禁止的是 score 矩阵的上三角。
- Vision Encoder 使用双向 self-attention，不能使用 causal mask。

### 3. Pre-Norm Transformer Block、MLP 与残差

核心手撕代码：

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, ratio=4):
        super().__init__()
        hidden_dim = d_model * ratio
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, ratio)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x
```

必须能解释：

- Pre-Norm 是 `LN -> 子层 -> Residual`。
- 两条残差路径始终保持 `[B,T,D]` 不变。
- MLP 只在最后一维变换：`D -> ratio*D -> D`，不会混合 token 位置。
- Attention 负责 token 间通信，MLP 负责每个 token 内部的特征变换。

### 4. Next-token prediction 与 label shift

当前代码由 `get_batch` 在数据层完成 shift，而不是在模型内部切片：

```python
def get_batch(data, batch_size, block_size):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y
```

模型 loss：

```python
logits = self.lm(hidden)                    # [B,T,V]
loss = F.cross_entropy(
    logits.reshape(-1, vocab_size),         # [B*T,V]
    targets.reshape(-1),                    # [B*T]
)
```

必须能解释：

```text
输入: [x0, x1, x2, x3]
标签: [x1, x2, x3, x4]
```

- Cross Entropy 在 vocabulary 维度分类。
- 当前字符数据没有 padding；未来加入 padding 后必须使用 `ignore_index=-100`。
- 如果数据层已经 shift，模型内部不能再 shift 一次，否则监督目标错位。

### 5. Patch Embedding

图像 `[B,C,H,W]` 被切成 patch token `[B,N,D]`：

```text
N = (H / P) * (W / P)
```

核心手撕代码：

```python
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, d_model):
        super().__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, image):
        x = self.projection(image)   # [B,D,H/P,W/P]
        x = x.flatten(2)             # [B,D,N]
        return x.transpose(1, 2)     # [B,N,D]
```

必须能解释：kernel 和 stride 都等于 patch size，因此 patch 不重叠；Conv2d 在这里等价于对每个展平 patch 做共享线性投影。

### 6. CLS token、位置编码与 ViT Block

当前 Vision Encoder 返回全部 token，供后续多模态模块使用：

```python
x = patch_embed(image)                         # [B,N,D]
cls = cls_token.expand(x.size(0), -1, -1)      # [B,1,D]
x = torch.cat([cls, x], dim=1)                 # [B,N+1,D]
x = x + pos_embed[:, :N + 1, :]                # [B,N+1,D]
for block in blocks:
    x = block(x)
x = final_norm(x)                              # [B,N+1,D]
```

ViT Block 与 TinyGPT Block 的核心差别：

```python
def forward(self, x):
    x = x + self.attn(self.ln1(x))  # 没有 causal mask
    x = x + self.mlp(self.ln2(x))
    return x
```

必须能回答：

- 为什么位置编码长度是 `N+1`：CLS token 也需要自己的位置。
- 输出为什么是 `[B,N+1,D]`：当前实现保留 CLS 和所有 patch token，而不是只做图像分类。
- 为什么 ViT 不用 causal mask：所有图像 patch 应互相可见。
- patch size 增大时，token 数按平方下降，计算更快但细节损失更多。

---

## 八、🟡 第二档：必须理解，不要求手撕完整代码

### TinyGPT

1. **完整 forward 流程**  
   Token Embedding + learned Position Embedding → Transformer Blocks → Final LayerNorm → LM Head。

2. **KV Cache**  
   必须能解释 prefill 和 decode：prompt 首次整体计算并缓存各层 K/V，之后每一步只输入新 token。当前 cache shape 为 `[B,H,T,Hd]`。应知道它减少重复计算，但不用背完整缓存代码。

3. **带 cache 的 causal mask 切片**  
   当前 query 长度可能为 1，而 key 长度为 `past_len+1`，mask shape 因而是 `[1,1,T_new,T_total]`。必须能定位 mask 切片错误，但不用默写接口。

4. **自回归生成**  
   理解 teacher forcing 与 generation 的区别；知道 temperature 越低越确定，top-k 只保留最高的 k 个候选。

5. **KV Cache 回退逻辑**  
   当前 learned position embedding 受 `max_seq_len` 限制。当 prompt + 新 token 超长时，生成函数回退到滑动窗口、每步重算。

6. **训练循环**  
   理解 `zero_grad -> backward -> step`、AdamW、train/eval 模式切换；不用背优化器 API。

7. **评估与 Perplexity**  
   `PPL = exp(mean cross entropy)`。理解为什么评测使用 `no_grad()` 和 `model.eval()`，以及评测后恢复训练模式。

8. **字符级 tokenizer**  
   理解词表小但序列长、未登录字符会报错，以及它与 BPE 在信息密度上的差别。

9. **Checkpoint**  
   知道模型权重、模型 config、`stoi/itos` 必须一起保存；不要求背 `torch.save` 格式。

### Vision Encoder

1. **ViT 的整体思想**  
   把图像变成 token 序列，再使用 Transformer encoder 建模 patch 间关系。

2. **CLS token 的作用**  
   它可作为全局图像表示；但 VLM 通常也需要 patch tokens，不能未经设计就只返回 CLS。

3. **learned position embedding**  
   当前位置编码绑定最大 patch 数；更换输入分辨率时通常需要插值或重新设计。

4. **Patch size 的计算权衡**  
   Self-attention 主要复杂度约为 `O(N²D)`。patch 越小，N 越大，视觉细节更多但注意力成本快速上升。

5. **输入图像预处理**  
   需要知道 resize、颜色通道和 normalize 会影响训练分布，但不用背 torchvision API。

6. **Vision Encoder 输出接口**  
   当前输出 `[B,N+1,D]`。后续 Image Projector 必须明确是使用全部 token、去掉 CLS，还是只取 CLS。

7. **初始化、dropout 与训练稳定性**  
   知道位置编码/CLS 初始化、attention dropout、MLP dropout 的作用；当前 TinyViT 是最小实现，没有这些增强。

### 必须会定位但不用手撕的常见 bug

- `softmax(dim=-2)`：归一化维度错误。
- causal mask 上下三角写反：未来信息泄漏或无法关注历史。
- label shift 两次：训练目标错一位。
- `transpose` 后直接 `view`：内存布局错误。
- CLS 拼接后仍使用 N 个位置编码：shape 不匹配。
- ViT 错加 causal mask：patch 无法双向交互。
- KV Cache 只拼 K、不拼 V：生成结果错误。
- cache 后位置 id 仍从 0 开始：每个新 token 都获得错误位置。

---

## 九、⚪ 第三档：可以不 care，需要时查询

- PyTorch 各函数的参数顺序和默认值。
- `Conv2d`、`Linear`、`LayerNorm` 的底层 CUDA kernel。
- FlashAttention 的 tiling、融合 kernel 与显存调度细节。
- AdamW、矩阵乘法和 softmax 的底层实现。
- checkpoint 序列化文件格式的内部细节。
- `pytest` fixture、benchmark API 和日志框架的具体写法。
- 各类现成 ViT/LLM 库的类名、配置字段和版本差异。
- 当前项目没有采用的复杂位置编码插值 API、分布式训练启动参数和生产部署配置。

第三档并不代表这些知识没有价值，而是面试准备的投入产出比低：知道它们解决什么问题即可，具体实现现场查。

---

## 十、当前模块 Review 结论

### TinyGPT

1. **输入 shape**：token ids `[B,T]`；带 KV Cache 时，每层历史 K/V 为 `[B,H,T_past,Hd]`。  
2. **输出 shape**：logits `[B,T,V]`；启用 cache 时额外返回每层 K/V。  
3. **核心公式**：causal self-attention + next-token cross entropy。  
4. **Mask**：下三角 causal mask；cache decode 时切成 `[1,1,T_new,T_total]`。  
5. **Loss**：当前由 `get_batch` 生成右移一位的 targets；所有字符 token 参与 loss。  
6. **已有测试**：MHA、TransformerBlock、TinyGPT 的 KV Cache shape 测试；旧 checkpoint 可兼容加载。

### Vision Encoder

1. **输入 shape**：图像 `[B,C,H,W]`。  
2. **输出 shape**：`[B,N+1,D]`，其中 `N=(H/P)*(W/P)`。  
3. **核心公式**：Patch Projection + 双向 self-attention + MLP。  
4. **Mask**：当前不使用 mask，所有 patch 双向可见。  
5. **Loss**：Vision Encoder 本身不计算 loss，监督信号由后续图文训练目标提供。  
6. **已有验证**：输入 `[2,3,32,32]` 时输出 `[2,17,64]`，forward 与 backward 均通过。

### 当前尚缺的测试

- TinyGPT causal mask 的“不能看到未来 token”数值测试。
- TinyGPT cache 与 no-cache logits 的逐位置一致性测试文件化。
- Vision Encoder 对不同 batch size、patch size 的 shape 测试。
- Vision Encoder 参数梯度和 CLS/position embedding 梯度测试。
- TinyGPT 与 Vision Encoder 的小样本 overfit test。

在补完这些测试前，可以认为核心 forward 已实现并能运行，但还未达到 `Memory.md` 所要求的完整模块验收标准。

---

## 十一、TinyVLM Base Caption Training 与视觉对齐验证

> 更新日期：2026-06-23  
> 本节记录 TinyVLM 在 Flickr8k 和 toy visual grounding 数据集上的训练、验证与图像消融结果。  
> 本阶段只验证 base image-caption training，不涉及 SFT、DPO、LoRA 或外部大模型。

### 1. Train / Validation 划分原则

Flickr8k 的每张图片对应 5 条 caption，因此不能直接随机划分 caption 行。否则同一张图片可能同时出现在训练集和验证集，产生数据泄漏。

当前实现按 **image filename** 划分：

```text
训练图片：7,282
验证图片：  809
训练 caption：36,410
验证 caption： 4,045
train / validation = 90% / 10%
seed = 42
```

必须理解：验证集的核心要求不是“caption 文本不同”，而是验证图片本身没有参与训练。

### 2. Caption 训练中的监督对齐

Dataset 构造：

```text
caption_ids = tokenizer.encode(caption)
input_ids = [BOS] + caption_ids
labels    = caption_ids + [EOS]
```

Padding 规则：

```text
input_ids padding = PAD token id
labels padding    = -100
```

`-100` 不能进入 token embedding。Dataset 只生成文本 labels；图像 token 对应的 `-100` 由 `TinyVLM.forward()` 内部补齐。

### 3. Flickr8k 训练结果

训练配置：

```text
GPU: RTX 5060 Laptop, 8GB
PyTorch: CUDA 12.8
image_size: 64
patch_size: 8
max_text_len: 96
batch_size: 64
learning_rate: 3e-4
epochs: 10
BF16 AMP: enabled
```

90/10 validation 训练结果：

```text
Epoch 1:  train_loss=2.3858, val_loss=2.1187
Epoch 5:  train_loss=1.5819, val_loss=1.5456
Epoch 10: train_loss=1.2952, val_loss=1.2926
```

最终验证指标：

```text
perplexity:          3.642
character accuracy: 60.94%
BLEU-1:              0.4266
BLEU-4:              0.0659
```

解释：character accuracy 是 teacher forcing 下的逐字符正确率，不能理解为 60.94% 的 caption 完全正确。BLEU-4 较低说明完整短语结构和语义准确性仍有限。

### 4. Image-shuffle、Zero-image 与 Noise-image Ablation

目的：判断模型是否真的使用 image tokens，而不是只利用 GPT 的语言先验。

四组验证输入：

```text
Normal:  图片、input_ids、labels 正常对应
Shuffle: input_ids 和 labels 不变，只在 batch 内随机打乱图片
Zero:    input_ids 和 labels 不变，图片替换为全零黑图
Noise:   input_ids 和 labels 不变，图片替换为固定种子的 [0,1] 均匀噪声
```

Flickr8k 最优模型在完整验证集上的结果：

```text
val_loss:         1.298494
val_shuffle_loss: 1.312771
val_zero_loss:    1.317246
val_noise_loss:   1.308113

shuffle_gap: +0.014277
zero_gap:    +0.018752
noise_gap:   +0.009619
```

结论：正常图片 loss 最低，说明模型使用了一部分视觉信息；但 gap 较小，语言先验仍占主导。Shuffle ablation 比 Zero/Noise ablation 更可靠，因为 shuffle 保持真实图片分布，而全零图和随机噪声都属于分布外输入。

### 5. Toy Visual Grounding Dataset

为了降低语言先验、直接检查视觉对齐，构造了颜色、形状、位置组合数据集：

```text
颜色：red / green / blue / yellow
形状：circle / square / triangle
位置：left / center / right
组合数：4 * 3 * 3 = 36
每种组合：30 个带尺寸、位置、背景扰动的样本
总图片数：1,080
训练图片：972
验证图片：108
```

Caption 完全由视觉属性决定，例如：

```text
image: 左侧红色圆形
caption: a red circle on the left
```

Toy 模型训练 30 epochs 后：

```text
train_loss:       0.0372
val_loss:         0.0369
val_shuffle_loss: 0.4024
val_zero_loss:    0.4647
val_noise_loss:   0.3062

shuffle_gap: +0.3655
zero_gap:    +0.4277
noise_gap:   +0.2693
```

正常、乱图、零图和噪声图的差距都非常明显，证明模型不是只记忆 caption 模板，而是在利用 image tokens。

### 6. Toy Visual Grounding 属性准确率

在 108 张未参与训练的验证图片上：

```text
color accuracy:    100.00%
shape accuracy:     56.48%
position accuracy:  94.44%
exact match:        52.78%
```

典型结果：

```text
target:    a red circle on the left
generated: a red square on the left
```

颜色和位置正确，但形状识别错误。当前模型已经产生明确视觉对齐，其中颜色最强、位置次之，circle / square / triangle 的轮廓区分仍是主要短板。

### 7. 当前阶段结论

1. TinyVLM 的 image → ViT → visual projection → GPT caption 链路能够正常训练。  
2. Flickr8k 上 normal loss 低于 shuffle/zero/noise loss，但差距较小，说明模型仍明显依赖语言先验。  
3. Toy grounding 上 shuffle/zero/noise loss 大幅上升，证明模型具备真实的视觉条件生成能力。  
4. 颜色和位置已经较好对齐，形状识别需要更强视觉表示、更多形状变化或更合适的 patch 设置。  
5. Zero/noise-image 结果必须结合 shuffle-image 解释，不能只依赖分布外输入得出结论。  

### 8. 相关代码与产物

```text
tinyvlm/train_vlm.py                  train/val、sample generation、ablation
scripts/run_image_ablation.py         Flickr8k 独立图像消融
tinyvlm/toy_grounding.py              toy grounding 数据生成
scripts/create_toy_grounding_dataset.py
scripts/evaluate_toy_grounding.py     属性准确率评估
checkpoints/best_tinyvlm_shuffle.pt
checkpoints/toy_visual_grounding.pt
logs/toy_grounding_metrics.json
logs/toy_grounding_ablation.json
logs/flickr8k_ablation.json
```

当前自动化测试：`29 passed`。前文“当前尚缺的测试”属于早期历史记录，其中 causal mask、cache 一致性、ViT 梯度、TinyVLM backward、dataset/collate、train/validation/generate/checkpoint 等测试现已补齐。

---

## 十二、TinyVLM Prefix Mask 与生成性能基准

> 更新日期：2026-06-23  
> 本节记录 TinyVLM 从全序列 causal mask 修改为 visual-prefix/text-causal mask 的实现与性能测试。

### 1. 为什么需要 Prefix Mask

旧实现对拼接后的完整序列直接使用下三角 causal mask：

```text
[image tokens, text tokens] -> full causal mask
```

这会导致前面的 image token 不能读取后面的 image token。虽然 TinyViT 输出本身已经经过双向 self-attention，但进入 GPT blocks 后，visual prefix 仍然受到不必要的因果限制。

新 Prefix Mask 规则：

```text
                  Image keys    Text keys
Image queries     双向可见       不可见
Text queries      全部可见       causal
```

矩阵结构：

```text
mask = [ image-image 全 1       image-text 全 0 ]
       [ text-image  全 1       text-text 下三角 ]
```

实现位于：

```text
tinyvlm/tinyvlm_model.py::build_prefix_mask
```

### 2. Prefix Mask Shape 示例

当 `N_img=2`、`T_text=3` 时：

```text
[
  [1, 1, 0, 0, 0],
  [1, 1, 0, 0, 0],
  [1, 1, 1, 0, 0],
  [1, 1, 1, 1, 0],
  [1, 1, 1, 1, 1],
]
```

前两行是 visual queries；后三行是 causal text queries。

### 3. 与 TinyGPT KV Cache 的关系

Prefix Mask 刚接入时的状态：

```text
TinyGPT standalone generate：默认使用 KV Cache
TinyVLM multimodal generate：当时暂未使用 KV Cache
```

因此 Prefix Mask 本身的修改：

- 不修改 TinyGPT 的纯文本 generate。
- 不修改 MultiHeadAttention 内部 K/V 拼接公式。
- 只修改 TinyVLM full forward 的 attention mask。
- 不要求同时修改 cache 的 K/V 拼接公式。

随后已经为 TinyVLM 实现多模态 prefill/decode：

```text
prefill cache shape: [B,H,N_img+T_prompt,Hd]
decode input:        仅新生成的 text token
decode mask:         [1,1,1,total_cached_length]
next position id:    N_img + T_prompt + generated_length
```

### 4. 修改前生成性能

测试条件：

```text
GPU: RTX 5060 Laptop 8GB
checkpoint: checkpoints/best_tinyvlm_shuffle.pt
new tokens: 32
repeats: 3
EOS early stop: disabled
KV Cache: disabled
```

初始实测：

```text
batch=1: 392.62 token/s, peak VRAM=13.47MB
batch=8: 3072.06 token/s, peak VRAM=17.39MB
```

### 5. Legacy Causal 与 Prefix Mask 公平对比

为了减少短任务的 GPU 波动，在同一 Python/CUDA 进程、同一模型、同一图片上切换 mask，每组重复 10 次：

```text
Legacy causal, batch=1: 413.62 token/s, peak=13.47MB
Prefix mask,   batch=1: 403.65 token/s, peak=13.47MB

Legacy causal, batch=8: 3027.94 token/s, peak=17.39MB
Prefix mask,   batch=8: 3087.54 token/s, peak=17.39MB
```

结论：

1. Prefix mask 没有增加峰值显存。  
2. 吞吐变化约为 `±2%`，属于正常运行波动。  
3. 两种 mask 的 attention 矩阵大小相同，复杂度仍为 `O(L²)`。  
4. Prefix mask 改善的是注意力语义，不是生成速度。  
5. 后续接入的 multimodal KV Cache 已显著提升逐 token 生成速度。  

### 6. Checkpoint 与测试状态

Prefix mask 不引入新参数，因此旧 checkpoint 的 `state_dict` 可以正常加载。但是旧 checkpoint 是在 full causal mask 下训练的，若要公平比较模型质量，需要在 Prefix Mask 下重新训练。

新增测试：

```text
tests/test_prefix_mask.py
```

测试覆盖：

- visual prefix 内部双向可见。
- visual queries 不能读取 text keys。
- text queries 可以读取全部 visual keys。
- text-to-text 保持 causal，不能读取未来文本。

### 7. Multimodal KV Cache 接入结果

当前 TinyVLM generation 默认启用 KV Cache，同时保留 `use_kv_cache=False` 作为数值验证和性能对照。

Prefill：

```text
images -> TinyViT -> visual projection
[visual embeddings, BOS/text prompt] -> Prefix Mask -> GPT blocks
每层缓存 K/V: [B,H,N_img+T_prompt,Hd]
```

Decode：

```text
input: 仅新生成的一个 text token
images: None，ViT 不再重复执行
position id: past_length
decode mask: [1,1,T_new,T_cache+T_new]
```

Decode mask 允许新文本 query 读取全部 cache，并在同一新 token chunk 内保持 causal。

验收测试：

- cached decode logits 与完整 multimodal forward 最后位置一致。
- cache/no-cache greedy generation token ids 完全一致。
- cached generation 中 ViT 只执行一次。
- 每层 K/V cache 随 decode token 正确增长。

RTX 5060 Laptop，固定生成 32 tokens，每组重复 10 次：

```text
No Cache, batch=1:  407.03 token/s, peak=13.47MB
KV Cache, batch=1:  787.90 token/s, peak=13.31MB

No Cache, batch=8: 3268.20 token/s, peak=17.39MB
KV Cache, batch=8: 6450.38 token/s, peak=16.00MB
```

固定生成 64 tokens，每组重复 5 次：

```text
No Cache, batch=1:  382.26 token/s
KV Cache, batch=1:  791.94 token/s

No Cache, batch=8: 3166.90 token/s
KV Cache, batch=8: 6627.98 token/s
```

当前实现获得约 `1.94x～2.09x` 加速。模型很小，因此 Python/kernel launch 开销仍占一定比例；随着模型和生成长度增加，避免重复计算的价值会更明显。

当前自动化测试：`29 passed`。

---

## 十三、按 11 项主线的最终验收记录

### 1. Attention / MHA / Causal Mask — ✅

代码：

```text
tiny_gpt/attention.py
tiny_gpt/multi_head_attention.py
tests/test_attention.py
```

已验证：MHA shape、下三角 mask、未来 value 不可见、`d_model % num_heads` 约束。必须能手写 `softmax(QKᵀ/√Hd)V` 和拆头/合头过程。

### 2. TransformerBlock — ✅

代码：

```text
tiny_gpt/transformer_block.py
tiny_gpt/feed_forward.py
vision_encoder/vit_block.py
```

当前实现为 Pre-Norm，两条 residual 均保持 `[B,T,D]`。GPT Block 接收 mask/cache；ViT Block 使用双向 attention。

### 3. TinyGPT Forward / Loss / Generate — ✅

代码：

```text
tiny_gpt/tiny_gpt.py
tiny_gpt/generate.py
tests/test_tiny_gpt_behavior.py
```

已验证：logits shape、next-token labels、loss backward、causal logits 不受未来 token 影响、最大位置长度保护、cached/uncached generation 一致。

### 4. KV Cache — ✅

代码：

```text
tiny_gpt/multi_head_attention.py
tiny_gpt/transformer_block.py
tinyvlm/tinyvlm_model.py
tinyvlm/multimodal_generate.py
tests/test_vlm_kv_cache.py
```

TinyGPT cache 保存纯文本历史 K/V；TinyVLM cache 保存 visual prefix 与文本 prompt 的 K/V。TinyVLM cached generation 中 ViT 只运行一次，32–64 token 生成获得约 `1.94x–2.09x` 加速。

### 5. ViT Patch Embedding / QKV / Block — ✅

代码：

```text
vision_encoder/patch_embedding.py
vision_encoder/vit_attention.py
vision_encoder/vit_block.py
vision_encoder/tiny_vit.py
```

已验证：`[B,3,H,W] -> [B,N,D] -> [B,N+1,D]`、fused QKV、CLS/position 参数梯度和 patch projection backward。

### 6. TinyVLM Bridge — ✅

代码：

```text
tinyvlm/tinyvlm_model.py
tests/test_tinyvlm.py
```

数据流：`ViT -> visual_proj -> GPT hidden space`。已通过 shape、loss、backward、单图 caption overfit 和 toy visual grounding。默认线性桥接；Q-Former-lite 替代方案见第十五节。

### 7. Image/Text Token 拼接 — ✅

核心：

```text
[image embeddings, text embeddings]
```

当前采用 visual-prefix/text-causal mask：image-image 双向、image-text 禁止、text-image 全可见、text-text causal。精确 mask 测试位于 `tests/test_prefix_mask.py`。

### 8. Text-only Loss Masking — ✅

代码：

```text
tinyvlm/dataset.py
tinyvlm/tinyvlm_model.py
tests/test_dataset_and_train.py
```

规则：`input_ids` padding 使用 PAD id；text labels padding 使用 `-100`；image token labels 由 TinyVLM forward 内部补 `-100`。`-100` 不进入 embedding。

### 9. VLM Generate — ✅

代码：

```text
tinyvlm/multimodal_generate.py
scripts/benchmark_vlm_generate.py
```

流程：image+BOS prefill，随后单 token cached decode，支持 greedy/sample、EOS 和 cache/no-cache 开关。Target caption 只用于评估打印，不进入生成输入。

### 10. SFT Answer-only Labels — ✅

实现文件（已统一在 `sft/` 包内）：

```text
sft/sft_dataset.py        # FlickerDataset（核心）
sft/sft_collator.py       # SFTCollator
sft/train_sft.py          # train_sft + evaluate_sft + load_pretrained
sft/test_sft_batch.py     # mask/shift/padding 单测
sft/test_sft_overfit.py   # 端到端 overfit
scripts/run_sft_train.py  # Flickr8k 入口
```

已完成 prompt/answer next-token 对齐（数据侧 shift）、prompt+padding labels=`-100`、answer-only loss、image prefix 由 TinyVLM 内部补 `-100`。详细 review 见 **第十四节**。模型复用 `tinyvlm/tinyvlm_model.py`，未重写。

### 11. DPO Logprob + Loss — ✅（数据/算法/骨架；训练未实跑）

```text
dpo/build_dpo_pairs.py  dpo/dpo_dataset.py  dpo/dpo_collator.py
dpo/dpo_loss.py（sequence_logprob + dpo_loss + freeze_ + dpo_step）
dpo/train_dpo.py（工程骨架）
```

answer-only sequence logprob、四组 logprob、reference freeze、`-logsigmoid` 符号、β 缩放均已实现并测试（`pytest dpo/` 12 passed）。详见第十六节与 `logs/dpo/dpo_notes.md`。剩训练循环实跑一轮。

### 下一阶段进入条件

```text
先能独立写出第 10 项的 answer-only labels
-> 再写 sequence answer logprob
-> 再写 DPO loss
```

不得跳过 SFT label mask 直接实现 DPO，因为 DPO 的 chosen/rejected logprob 依赖完全相同的 answer mask 与 next-token 对齐。

---

## 十四、SFT Answer-only Labels（模块四）

> 更新日期：2026-06-24
> 数据集核心（FlickerDataset）由人工手写，工程件（collator/config/train/test）与实验分析由 AI 补全并经本节 review。模型复用 `tinyvlm/tinyvlm_model.py`，未重写。

### 文件位置（已统一到 `sft/` 包）

```text
sft/sft_dataset.py        # FlickerDataset（核心，已从仓库根目录挪入）
sft/sft_collator.py       # SFTCollator：变长 padding + label/attn mask + image stack
sft/train_sft.py          # train_sft 循环 + evaluate_sft（图像消融）+ load_pretrained
sft/test_sft_batch.py     # mask / shift / padding 单测（5 个）
sft/test_sft_overfit.py   # 端到端 overfit 冒烟测试
scripts/run_sft_train.py  # Flickr8k 入口
```

### 6 个必答问题

**1. 输入 shape**：`base_dataset[i] -> (image[3,H,W], caption:str)`；`__getitem__ -> input_ids[T], labels[T]`（变长 T=P+A-1）；`SFTCollator -> input_ids/labels/attention_mask[B,Tmax], image[B,3,H,W]`。

**2. 输出 shape**：`TinyVLM.forward(image,input_ids,labels) -> logits[B, N_img+T, V], loss 标量`，`N_img=(img/patch)^2+1` 含 CLS。

**3. 核心公式（next-token CE + response-only）**：

```text
full      = prompt_ids + answer_ids(含 EOS)
input_ids = full[:-1]          # 数据侧左移
labels    = full[1:]
labels[: len(prompt_ids)-1] = -100
loss = CE(logits, full_labels, ignore_index=-100)   # forward 内 full_labels 前补 N_img 个 -100
```

关键：本仓库模型 forward **不 shift**（`tiny_gpt.py` / `tinyvlm_model.py` 都是 `CE(logits[t], labels[t])`），shift 必须在数据侧做，与 `get_batch` / `ImageCaptionDataset` 一致。

**4. mask 在哪用**：causal/prefix mask（模型内，visual 双向 + text causal）；padding mask（collator 的 attention_mask，SFT 走右 padding+causal 故未显式用）；label mask `-100`（prompt+padding+visual prefix 都不算 loss）。

**5. loss 算哪些 token**：只算 answer token + 末尾 EOS；prompt、padding、visual prefix 全 `-100`。

**6. 测试**：`test_sft_batch.py` 5 个（prompt 段 -100 / answer 段==answer_ids含EOS / padding 段 -100 / 被监督数==answer 长度 / CE 直接对齐且全 -100 触发 NaN）；`test_sft_overfit.py` 端到端 overfit loss 大幅下降；`evaluate_sft` 图像消融。`sft/` 共 6 passed。

### 核心问题：label shift 约定反了（本次最大的坑）

现象链：SFT 训练 loss 看着漂亮地降到接近 0，但图像打乱消融发现「正常图 / 乱图 / 黑图」的 loss 几乎相等，且生成退化成单字符重复——说明模型根本没用图像。

根因：本仓库的 next-token shift 约定是 **在数据侧做、模型 forward 不做**（`tiny_gpt.py` / `tinyvlm_model.py` 都是 `CE(logits[t], labels[t])`，由 `get_batch` / `ImageCaptionDataset` 在数据里左移 label）。而 FlickerDataset 最初写成 `labels = input_ids.clone()`（"模型负责 shift" 的相反约定），等于训练模型「用 token t 预测它自己」——一个恒等映射，因果隐状态已含当前 token，预测自己 trivial，loss 当然假降，图像/视觉桥接完全用不上。

为什么难发现：这是 silent bug，mask 位置单测全过（prompt=-100、answer 保留都对），只有跑真实训练 + 图像消融才暴露。

修复：数据侧左移——`input_ids = full[:-1]`，`labels = full[1:]`，再把前 `len(prompt_ids)-1` 个 label 置 -100。

教训：**teacher-forced loss 会骗人**；验证多模态是否真看图，必须做图像打乱/置零消融（`normal << shuffled` 才算 grounding），不能只看 loss 下降。

### 其它已修 bug

1. `def FlickerDataset(Dataset)` 写成函数而非 `class` -> NameError，不可用。
2. answer 末尾缺 `EOS` -> 模型学不到停止；已在末尾加 `eos_token_id`。
3. 截断在"完整序列"上 `[:max_len]` -> 超长样本 EOS 被截掉、极端长 prompt 整条 -100 -> NaN。改为分段截断：先压 prompt 给 answer 留位，截 answer 时把 EOS 接回。

### Conclusion

通过全部 mask/shift 单测与端到端 overfit；next-token 对齐与 answer-only loss 已修正为与仓库约定一致，可进入 DPO（其 chosen/rejected logprob 复用完全相同的 shift 与 answer mask）。Q-Former-lite 桥接已手写完成，见第十五节（线性 `visual_proj` 仍是默认）。

---

## 十五、Q-Former-lite 视觉桥接（模块三）

> 更新日期：2026-06-24
> `CrossAttention` / `QFormLiter` 由人工手写，集成 glue（`TinyVLMQFormer` / `build_qformer_model`）与对比实验由 AI 补全。实验数据见 `logs/sft/sft_experiments.md`（实验三），本节只记 review/问题。

### 文件

```text
tinyvlm/qformer.py   CrossAttention（手写多头 cross-attn 块）+ QFormLiter（可学习 query + 多层）
                     + TinyVLMQFormer / build_qformer_model（接进 TinyVLM 替换线性 visual_proj）
```

### 6 个必答问题

**1. 输入 shape**：`image_tokens [B, N, img_dim]`（N = patch + CLS，来自 TinyViT）。
**2. 输出 shape**：`[B, Q, d_model]`，Q = `num_query_token`，**与 N 解耦**（验证：N=17→Q=16、N=50→仍 16）。
**3. 核心公式**：cross-attention `softmax(QKᵀ/√head_dim)·V`，**Q 来自可学习 query token、K/V 来自 image token**，score 形状 `[B,H,Q,N]`（长方）。
**4. mask**：无——每个 query 看全部 image token（image 都有效，不像 decoder 要 causal）。
**5. loss**：Q-Former 自身不算 loss；它产出 visual prefix，loss 在 TinyVLM 的 response-only CE 里（visual prefix 段补 -100）。
**6. 验证**：shape 测试（输出 Q 个、换 N 不变）、backward（query_token 有梯度）、集成进 TinyVLM forward（logits `[B, Q+T, V]`）、接进 SFT 训练跑通。

### Bugs found（review 中修掉）

1. cross-attn 最初用 `nn.MultiheadAttention`（torch 封装）-> 违反"核心手写"，改成手写 `q/k/v/out_proj + QKᵀ/√d`。
2. `QFormLiter` 一批写错：`super().__init()`、`for _ in num_layers`（int 不可迭代）、`nlp_ratio`（应 `mlp_ratio`）、`nn.LayerNrom`、`query_token` 单复数不一致、`self.final`（应 `self.final_lm`）——逐一修复后跑通。

### 两个概念要点

1. **cross vs self**：自注意力 Q/K/V 同源、score 方阵、长度不变；cross-attn 两个输入、score 长方 `[Q,N]`、输出 Q 个——把 N 个 patch **重采样成固定 Q 个**视觉 token，是 Q-Former 的灵魂。
2. **粒度**：`CrossAttention` 内含 LN+attn+残差+FFN，其实是一个完整 block（对标 `TransformerBlock`/`ViTBlock`，不是 `MultiHeadAttention`）；命名上更准应叫 `QFormerBlock`。

### 集成与约束

`TinyVLMQFormer` 子类化 TinyVLM、把 `visual_proj` 换成 `QFormLiter`。TinyVLM.forward 里 `image_length` 按桥接输出长度动态取，所以 N→Q 后 prefix mask / label 补 -100 / 拼接全自动适配，**未改 TinyVLM**。约束：桥接输出维度 `d_model` 必须 == `gpt_dim`。

### Conclusion

shape / 集成 / SFT 训练均验证通过。对比实验（实验三，32px 与 64px 两轮）显示：**从零训练 + tiny 模型 + 小数据下，Q-Former 并不优于线性桥接**——32px（N≈Q）两者相当；64px（N=65≫Q=16）线性桥接 grounding 反而更强（它把全部 patch 原样喂给 LLM，而 Q-Former 的 16 query 瓶颈在无 query 预训练时反而丢信息）。Q-Former 的真实价值是规模化效率（prefix 不随分辨率膨胀）与 BLIP-2 式大规模 query 预训练，本 toy 设定喂不出。遗留：类名 `QFormLiter`/`final_lm` 可统一；若要复现 Q-Former 优势需引入 query 预训练。

---

## 十六、多模态 DPO（模块五）

> 更新日期：2026-06-25
> 本轮完成数据工程 + 核心算法 + 工程骨架；训练循环可跑但尚未实跑一轮。详细记录见 `logs/dpo/dpo_notes.md`。

### 文件

```text
dpo/build_dpo_pairs.py   SFT json -> {image,prompt,chosen,rejected}（rejected=随机别条 caption）
dpo/dpo_dataset.py       chosen/rejected 各自 prompt+answer token + 对齐 label（prompt=-100）
dpo/dpo_collator.py      chosen/rejected 分别右 padding，image stack（两边共用）
dpo/dpo_loss.py          sequence_logprob + dpo_loss（核心，手写）+ freeze_ + dpo_step（glue）
dpo/train_dpo.py         policy(从 SFT 初始化) + reference(deepcopy+freeze) + 训练循环（骨架）
```

### 核心数学

```text
(A) answer-only 序列 logprob：log π(y|x) = Σ_{t∈answer} log softmax(logits_t)[y_t]
(B) DPO loss：L = -E[ log σ( β·((logπθ_w-logπref_w) - (logπθ_l-logπref_l)) ) ]
    隐式 reward r(y)=β·(logπθ(y)-logπref(y))；chosen 更被偏好 -> 括号>0 -> loss小（符号别反）
```

### shift 约定（与 SFT 的关键区别）

DPO 用「对齐 label + 在 logprob 里 shift」：`input_ids = prompt_ids + answer_ids`（完整），`labels` 同位对齐（prompt 段 -100），`sequence_logprob` 自己做 `logits[:, N_img-1:N_img+T-1]` 的左移。区别于 SFT FlickerDataset 的「数据侧 shift」——因为 DPO 不走模型内置 cross_entropy，而是自己 gather logprob。`N_img` 从 `logits.size(1)-T` 反推，线性桥接 / Q-Former 都通用。

### Bugs found（review 中修掉，均为 typo 级）

`sequence_logprob`：切片 `N_img-1:N_img+1`（长度仅 2）应为 `N_img-1:N_img+T-1`；`token_log`→`token_logp`；`.sun`→`.sum`。
`dpo_loss`：`policy_chosen_logs`→`policy_chosen_logps`；`ref_chose_logs`→`ref_chosen_logps`。metrics 改存 `.item()`。

### 测试

`pytest dpo/` 12 passed（6 数据 + 6 loss）。loss 覆盖：shape/≤0、`policy==ref -> loss==log2`（精确锚点）、符号方向、β 缩放、梯度只进 policy（reference 全 None）。

### Conclusion

DPO 数据/logprob/loss/骨架就绪且测试通过。遗留：`train_dpo.py` 实跑一轮（policy 从 SFT ckpt 初始化，盯 reward_accuracy/margin）；rejected 升级 hard negative；DPO 前后幻觉对比评测。
