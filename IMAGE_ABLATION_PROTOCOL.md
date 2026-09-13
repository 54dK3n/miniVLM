# Image-Shuffle Ablation：动机、干预设计与两组数字的可比性核查

本文档回答三件事：

1. **我们为什么要做这个实验** —— 它要排除的是哪一种"假成功"；
2. **我们具体改了什么** —— 干预前图片是什么、干预后换成了什么（真图？黑图？纯噪声？），文字侧动没动；
3. **Flickr8k 的 `+0.0143` 和 synthetic 的 `+0.3655` 在流程上哪些一致、哪些不一致**，写进文书时必须设哪些限。

数字来源：`logs/flickr8k_ablation.json`、`logs/toy_grounding_ablation.json`，
对应 `REVIEW_LOG.md` 第 783–847 行的实验记录。

---

## 1. 动机：loss 下降不能证明模型在看图

TinyVLM 的训练目标是 teacher forcing 下的 next-token 交叉熵：模型在预测第 `t` 个字符时，
**前 `t-1` 个字符是真实答案直接喂进去的**。这带来一个致命的歧义：

- caption 本身有极强的语言结构。本项目用的是 char-level tokenizer（vocab 只有 47），
  在已经看到 `"A dog is runn"` 之后预测 `"i"`，纯语言模型也能做得很好；
- 于是即使视觉通路完全废掉，训练/验证 loss 一样会一路下降，BLEU 也会有非零分数。

具体的失败模式（在这个架构里都可能发生，而且从 loss 曲线上完全看不出来）：

- `visual_proj`（`tinyvlm/tinyvlm_model.py:45`）权重塌缩，输出近似常数；
- TinyViT 输出与输入图片几乎无关（例如所有图片映射到同一个向量）；
- GPT 的 attention 直接忽略前 65 个 visual prefix token，只用文本上下文；
- 早期还踩过一类更隐蔽的坑：**label shift 约定弄反**，监督目标错位，模型退化成
  「复制输入」，此时 loss 也能降但 gap 恒为 0（这个坑发生在 SFT 那条链路上，
  见 `REVIEW_LOG.md:1272`，不是本文这条 captioning 链路，但正是它让我们确信
  「只看 loss 不够，必须有独立的 grounding 探针」）。

所以需要一个**不依赖 loss 绝对值**的判据：

> 主动破坏「图片 ↔ 文字」的对应关系，看 loss 涨多少。
> 涨得多 ⇒ 模型原本确实从图片里取了信息；几乎不涨 ⇒ 图片没被用上。

这就是 image ablation 的全部逻辑。它测的是 **image 和 text 之间的互信息被模型用掉了多少**，
单位是 nats/token。

### 1.1 三个最容易搞混的点（先讲清楚）

1. **干预发生在 validation 阶段，不在 training。**
   训练全程都是正确配对，从来没有打乱过训练数据。消融是：拿训练好的 checkpoint，
   在验证集上多跑几遍前向，每遍换一种图片输入，比较 loss。模型权重不受任何影响。

2. **不要和 DataLoader 的 `shuffle=True` 混淆。**
   `train_vlm.py` 里 `train_loader` 的 `shuffle=True` 是普通的**样本顺序打乱**（标准训练做法），
   `val_loader` 则是 `shuffle=False`。这与消融里的图文错配是两回事。
   checkpoint 文件名 `best_tinyvlm_shuffle.pt` 里的 "shuffle" 指的是前者，不是消融。

3. **`toy` 与 `zero`/`noise` 不是同一层面的东西。**
   `zero`/`noise` 是**同一套消融里的另外两个 condition**（换图片输入）；
   `toy` 是**换了一个数据集**（换任务难度和语言先验强度）。

### 1.2 实验实际发生的顺序（按文件时间戳）

| 时间 | 事件 | 当时的消融路数 |
|------|------|---------------|
| 06-23 19:13 | Flickr8k 模型训完（`best_tinyvlm_shuffle.pt`） | normal + shuffle |
| — | 看到 `shuffle_gap ≈ +0.014`，**这时才出现歧义**：模型没看图？还是看了但增益本就很小？ | |
| 06-23 19:32 | toy 数据集 + toy 模型训完（`toy_visual_grounding.pt`） | normal + shuffle + zero |
| 06-23 19:33 | `logs/toy_grounding_metrics.json`（含属性准确率） | 同上 |
| 06-23 23:18 | 加上 noise 一路，**两个数据集用同一版代码各重跑一次** | 四路齐全 |
| | → `logs/flickr8k_ablation.json`、`logs/toy_grounding_ablation.json` | |

所以现在引用的 `+0.0143` 与 `+0.3655` **是同一版代码、同一时间点产出的**，这一点是可比的；
不可比的部分见第 8 节。

注意动机顺序：做这个探针**不是**因为事先认定模型没看图，而是因为 loss 本身在原理上
无法证伪"没看图"。是探针跑出小 gap 之后，才需要第二个实验（toy）去消解歧义。

---

## 2. 干预前后，"图片"到底是什么

### 2.1 干预前（normal）：真实配对的那张照片

数据管线（`tinyvlm/dataset.py`）：

```
captions.txt/csv 每行 = (image_name, caption)
   ↓ _load_image (dataset.py:70)
PIL 打开 → convert("RGB") → resize 到 64×64 (bilinear) → /255.0
   ↓
float32 张量 [3,64,64]，值域 [0,1]，**没有做 ImageNet mean/std normalize**
   ↓ TinyViT(patch_size=8)  (train_vlm.py:22,34)
(64/8)² + 1 = 65 个 visual token，每个 64 维
   ↓ visual_proj: Linear(64→64)
   ↓ 与文本 embedding 拼接：[65 个 image token][T 个 text token]
   ↓ prefix mask：visual 段内部双向、文本段因果、文本可看全部 visual
   ↓ TinyGPT (d_model=64, 4 heads, 2 layers)，整模型 0.237M 参数（ViT 0.117M）
   ↓ cross_entropy(ignore_index=-100)
loss 只在文本位置计算（image 那 65 个位置的 label 被强制填 -100，tinyvlm_model.py:148）
```

文本侧：`input_ids = [BOS] + caption_ids`，`labels = caption_ids + [EOS]`
（shift 在数据层完成，`dataset.py:86-89`；模型内部**不再** shift）。

**所以"干预前"这一路里，喂进去的图片就是这条 caption 真正对应的那张原图，是真配对。**
它就是 `val_loss`，两个数据集分别是 1.2985 和 0.0369。

### 2.2 干预后：三种替换，文字一个 token 都不动

三种干预**只替换 `images` 这一个张量**，`input_ids` 和 `labels` 完全保持原样：

| 条件 | 喂进去的图片是什么 | 是不是自然图片 | 破坏了什么 | 代码 |
|------|-------------------|---------------|-----------|------|
| **normal** | 这条 caption 真正对应的原图 | 是 | —（基线） | `train_vlm.py:100` |
| **shuffle** | **同一个 batch 里另一条样本的真实照片** —— 仍然是真图、同样的 64×64 预处理、同样的像素统计，只是配错了对象 | **是** | 只破坏图文配对（互信息），不改变图片的边缘分布 | `train_vlm.py:101` |
| **zero** | `torch.zeros_like(images)` → **纯黑图**，RGB 三通道全 0（值域下界），不是噪声 | 否（分布外） | 配对 + 图片本身信息量全部抹掉 | `train_vlm.py:102` |
| **noise** | `torch.rand(...)` → 每个像素独立同分布 **uniform[0,1] 的彩色雪花**（不是高斯噪声，也没有任何空间结构） | 否（分布外） | 同上，但保留了"输入非常值"这一点 | `train_vlm.py:103-109` |

三点必须在文书里写清楚，否则读者会误解：

1. **shuffle 用的是真实照片，不是噪声。** 这是它和 zero/noise 的本质区别（见第 3 节）。
2. **打乱范围是一个 batch（≤64 张），不是整个 validation set。**
3. **打乱是普通 `randperm`，不是 derangement**，允许某个样本映射到自己；
   只有当整个 batch 恰好抽到恒等置换时才 `roll(1)` 兜底（`train_vlm.py:90-93`）。
   逐元素的自映射不做处理 —— 实测泄漏率见第 8.2 节。

### 2.3 真实例子（用 `ablation_seed=0` 精确重放出来的第一个 batch）

Flickr8k：

| 行 | caption（不变） | 原图 | shuffle 后喂进去的图 | 那张图自己的 caption |
|---|---|---|---|---|
| 0 | `A brown and white dog is running through the snow .` | `101654506_….jpg` | `104136873_….jpg` | `Three people hang out on top of a big hill .` |
| 1 | `A dog is running in the snow` | `101654506_….jpg` | `101654506_….jpg` | ← **自映射，这一行其实没被打乱** |
| 2 | `A dog running through snow .` | `101654506_….jpg` | `109738916_….jpg` | `Two helmeted men sit on yellow snowmobiles …` |
| 3 | `a white and brown dog is running through a snow covered field` | `101654506_….jpg` | `111497985_….jpg` | `The person in the striped shirt is mountain climbing .` |

（前四行 caption 属于同一张图 —— Flickr8k 每张图 5 条 caption，在文件里连续排列，
`shuffle=False` 的 DataLoader 会把它们放进同一个 batch，这直接导致了第 8.2 节的泄漏。）

Toy synthetic：

| 行 | caption（不变） | 原图 | shuffle 后喂进去的图 | 那张图自己的 caption |
|---|---|---|---|---|
| 0 | `a yellow circle on the center` | `toy_00867.png` | `toy_00325.png` | `a green circle on the center`（同形状同位置，只有颜色不同） |
| 1 | `a green triangle on the right` | `toy_00537.png` | `toy_00537.png` | ← **自映射，没被打乱** |
| 2 | `a yellow square on the right` | `toy_00972.png` | `toy_00539.png` | `a green triangle on the right` |
| 3 | `a yellow triangle on the center` | `toy_01038.png` | `toy_01002.png` | `a yellow triangle on the left`（只有位置不同） |

可以看到 toy 这边即使打乱成功，新图片也常常和原 caption **部分属性重合**
（只有 4 色 × 3 形 × 3 位置 = 36 种组合），这会系统性地**压低** toy 的 gap。

---

## 3. 为什么要三路干预，而不是只做一种

zero 和 noise 都是**分布外输入**：模型训练时从没见过纯黑图或均匀雪花。
它们的 loss 上升可能来自两个完全不同的原因：

- (a) 真的丢失了配对的视觉信息 —— 这是我们想测的；
- (b) 仅仅因为输入异常，导致 ViT 激活跑到训练时没覆盖的区域 —— 这是伪信号。

**shuffle 不存在这个混淆**：图片仍是训练分布内的真实照片，像素统计、ViT 激活分布都正常，
唯一被破坏的就是「这张图 ↔ 这句话」的对应关系。因此：

> **shuffle 是唯一干净的因果对照，zero/noise 只作为辅助上下界参考。**
> `REVIEW_LOG.md:809` 里写的就是这个结论。

三者一起看还能区分「用了多少级别的视觉信息」。实测：

| | noise gap | shuffle gap | zero gap | 解读 |
|---|---|---|---|---|
| Flickr8k | +0.0096 | **+0.0143** | +0.0188 | 三者都很小；模型只用到了很浅的视觉信息 |
| Toy | +0.2693 | **+0.3655** | +0.4277 | 三者都很大；模型确实在读图 |

顺序都是 `noise < shuffle < zero`，符合预期：随机噪声图至少还提供"有东西"的激活，
而全黑图连这个都没有。如果出现 `shuffle ≈ 0` 但 `zero` 很大，那就说明模型只用了
"图片存不存在/整体亮度"这种与内容无关的信息 —— 这正是需要 shuffle 才能识别的情况。

---

## 4. 代码实现（两个数据集共用同一份）

入口函数：`tinyvlm/train_vlm.py:64` 的 `validate()`。
它同时被训练循环（`train_vlm.py:275`，每个 epoch 打印四路 loss）和两个独立评估脚本调用。

```python
@torch.no_grad()
def validate(model, data_loader, device, amp_enabled=False,
             amp_dtype=torch.float16, ablation_seed=0):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(ablation_seed)   # 默认 0

    for images, input_ids, labels in data_loader:
        ...
        valid_tokens = labels.ne(-100).sum().item()          # 只数真实文本 token

        permutation = torch.randperm(images.size(0), device=device, generator=generator)
        if images.size(0) > 1 and torch.equal(permutation,
                torch.arange(images.size(0), device=device)):
            permutation = permutation.roll(1)                # 整批恒等时兜底

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            _, loss         = model(images,                   input_ids, labels)  # normal
            _, shuffle_loss = model(images[permutation],      input_ids, labels)  # shuffle
            _, zero_loss    = model(torch.zeros_like(images), input_ids, labels)  # 纯黑
            noise_images    = torch.rand(images.shape, dtype=images.dtype,
                                         device=device, generator=generator)
            _, noise_loss   = model(noise_images,             input_ids, labels)  # 雪花

        total_loss += loss.item() * valid_tokens             # 按有效 token 数加权
        ...
    return (total_loss / total_tokens, ...)                  # per-token 平均 CE
```

细节：

- 四路在**同一次循环、对同一批张量**计算，不存在数据加载顺序、dropout、AMP 设置差异；
- `F.cross_entropy(..., ignore_index=-100)`（`tinyvlm_model.py:155`）返回的是 batch 内
  有效 token 的平均，外层再按 `valid_tokens` 加权求和 → 最终是**全验证集的 per-token 平均 CE（nats）**；
- image 位置的 label 恒为 `-100`，所以替换图片**只通过 attention 影响文本位置的预测**，
  不会因为"图片 token 变多变少"改变分母；
- gap 的定义就是 `shuffle_loss − val_loss`，两边完全一致。

调用点：

| 数据集 | 脚本 | 行 | 是否传 `ablation_seed` | batch_size |
|---|---|---|---|---|
| Flickr8k | `scripts/run_image_ablation.py` | 61-63 | 否（用默认 0） | 默认 64 |
| Toy | `scripts/evaluate_toy_grounding.py` | 69-71 | 否（用默认 0） | 默认 64 |

两个脚本都用 `split_by_image(val_ratio=0.1, seed=42)` 按**图片**划分训练/验证，
保证同一张图的多条 caption 不会跨越 split（`dataset.py:124`）。

---

## 5. 第二次改动：为什么又造了一个合成数据集

Flickr8k 上跑完第一版消融，结果是 `shuffle_gap = +0.0143`（相对 +1.10%）。
这个数字有歧义 —— 它可能意味着：

- (i) 模型确实几乎没在看图；也可能
- (ii) 模型在看图，但 Flickr8k 的 caption 语言先验太强，0.237M 参数的模型又太小，
  以至于「看图」带来的收益本来就只值 0.01 nats。

**只用 Flickr8k 无法区分这两种解释**，于是做了第二次改动：造一个语言先验被压到最低的
toy grounding 数据集（`tinyvlm/toy_grounding.py`）：

```
4 色 × 3 形状 × 3 位置 = 36 种组合，每种 30 张（尺寸/位置/背景亮度有扰动）
共 1080 张 64×64 图，caption 完全由视觉属性决定：
   "a red circle on the left"
词表极小、句式唯一 —— 不看图就只能瞎猜，语言先验的天花板被人为压死
```

结果 `shuffle_gap` 从 `+0.0143` 跳到 `+0.3655`，配合属性准确率
（颜色 100%、位置 95.4%、形状 56.5%、exact match 52.8%）。

**这个结果的正确解读（很容易说过头）：**

✅ 能直接下的结论：
- 视觉通路是**通的** —— 排除了 `visual_proj` 塌缩、ViT 输出恒定、attention 忽略 visual prefix、
  label shift 错位这一类硬故障。这是 toy 实验最主要的价值：它证伪的是「代码/架构坏了」。
- 在语言先验被压死的任务上，这套 ViT→projection→GPT 链路**能够** grounding。
- toy 内部唯一可直接断言的能力短板是**形状判别**（56.5%，而颜色 100%、位置 95.4%）——
  弱在轮廓区分，不是弱在整个视觉通路。

❌ 不能下的结论：
- 不能说"Flickr8k 上 gap 小是因为视觉能力太弱"。toy 相对 Flickr8k **同时**变了四件事：
  数据难度（合成色块 vs 自然照片）、语言先验强度（36 种句子 vs 4 万条自由文本）、
  训练时长与学习率（30 epoch/1e-3 vs 10 epoch/3e-4）、caption 长度（32 vs 96）。
  这四个因素在 toy 实验里**完全混淆**，没有被分离。

因此 Flickr8k 那一侧最多只能表述为：

> 在「自然图像 + 强语言先验 + 0.237M 参数 + 10 epoch」的组合下，
> 视觉信息带来的边际收益只有约 0.014 nats/token（相对 +1.1%）；
> 该值显著大于数值噪声底（~1e-4），因此视觉通路确实在起作用，只是贡献很小。

要把"容量不足"和"语言先验太强"分开，需要额外的受控实验（例如在 Flickr8k 上放大模型、
或在 toy 上刻意加入语言先验），当前两组数字做不到这一点。

---

## 6. 预期结果 vs 最终结果

### 6.1 Flickr8k：预期与实际

| 项目 | 做之前的预期 | 实际结果 | 判定 |
|------|-------------|---------|------|
| shuffle gap | 若模型真在看图，打乱后 loss 应明显上升，量级预期 0.1~1.0 nats（相对 ≥10%） | **+0.0143（+1.10%）** | ❌ 比预期低一个数量级 |
| 是否显著 | 至少要明显超过数值噪声 | 噪声底 ~1e-4，gap 是其 ~140 倍 | ✅ 显著，但很小 |
| 三路排序 | `normal < shuffle < noise ≈ zero`（OOD 输入应当更差） | normal 1.2985 < **noise 1.3081 < shuffle 1.3128** < zero 1.3172 | ⚠️ noise 反而比 shuffle 低 |

第三行值得单独解释：如果模型真的在读**图片内容**，那么「一张配错的真实照片」应当比
「一团没有任何内容的均匀雪花」更接近正常输入、loss 更低。实测却相反（shuffle 比 noise 高 0.0047）。
这说明在 Flickr8k 上，四种输入之间的 loss 差异更像是**激活幅度/分布扰动**造成的，
而不是"看到了什么内容"的差别 —— 本身就是弱 grounding 的旁证。

**当时漏掉的对照**：没有训练一个**纯文本 LM** 作为 baseline，所以"1.2985 比纯语言先验好多少"
无法直接回答。最接近的代理是 zero-image loss 1.3172（模型完全拿不到视觉信息时的表现），
二者只差 1.4%。

### 6.2 toy：预期与实际（有可计算的理论参考线）

toy 的 caption 完全由 36 种属性组合决定，因此"完全看不到图时的最优 loss"是可以**算出来**的：
一个恰好知道 caption 分布、但看不见图的 oracle 语言模型，在该验证集上的 per-token CE 为

```
验证集：108 条 caption / 36 种 / 平均 28.3 token（含 EOS）
oracle prior-only CE = log(36) / 28.3 ≈ 0.1232 nats/token
```

以这条线为参照：

| 条件 | 预期 | 实际 | 相对 0.1232 参考线 |
|------|------|------|-------------------|
| normal | 若在看图 → 应远低于 0.1232 | **0.0369** | 0.30×　✅ 符合预期 |
| shuffle | 朴素预期：退回"无知"水平 ≈ 0.1232 | **0.4024** | **3.3×　超出预期** |
| zero | ≥ 参考线 | 0.4647 | 3.8× |
| noise | ≥ 参考线 | 0.3062 | 2.5× |

最关键的一条：**shuffle 后的 loss 不是回落到 prior 参考线，而是冲到它的 3.3 倍。**
含义是模型不会在图片不可信时退回语言先验 —— 它**相信了那张错误的图**，
于是比干脆忽略图片还要差 3 倍。这比 gap 的绝对数值本身是更强的 grounding 证据；
同时也暴露一个能力缺失：模型没有任何"图文不一致"的检测能力（它从没被训练过这种输入）。

### 6.3 属性准确率：预期与实际

| 属性 | 随机基线 | 预期 | 实际 |
|------|---------|------|------|
| 颜色 | 25% | 高（颜色是最低级的像素特征） | **100%**　✅ |
| 位置 | 33% | 高（patch 顺序 + position embedding 直接携带） | **95.4%**　✅ |
| 形状 | 33% | 中高 | **56.5%**　❌ 只比随机好 23 个点 |
| exact match | ~2.8% | — | **52.8%**（几乎完全由形状决定） |

形状是唯一明显不及预期的一项，而且原因是可定位的：图中形状半径 8–10 px（直径 16–21 px），
而 `PATCH_SIZE = 8`，一个形状只覆盖约 2×2~3×3 个 patch，圆/方/三角的轮廓差异在
patch 化 + 只有 2 层的 ViT 之后基本被抹平。这是一条具体、可行动的结论
（缩小 patch、放大输入分辨率、或加深 ViT），而不是笼统的"视觉能力弱"。

### 6.4 最终结论（三句话）

1. **Flickr8k**：视觉通路确实在起作用（gap 显著高于噪声底），但贡献极小
   —— +0.0143 nats/token，相对 +1.10%，语言先验占绝对主导；且四路 loss 的排序显示
   模型基本没把图片当作**内容**特征使用。
2. **toy**：把语言先验压死之后，同一套架构表现出强 grounding
   （normal 只有 prior 参考线的 0.30×，shuffle 反冲到 3.3×，颜色/位置接近满分）
   —— 这证伪了"代码或架构坏了"，Flickr8k 上的小 gap 不是硬故障。
3. **仍未解决**：Flickr8k 那边的瓶颈究竟是模型容量（0.237M）、训练时长（10 epoch）、
   还是自然图像本身太难，两组实验把这些因素混在一起，无法区分；且缺少纯文本 LM 对照。
   唯一被 toy 直接定位的短板是**形状/轮廓判别**。

---

## 7. 四点一致性核查（你问的四个问题）

| # | 问题 | 回答 |
|---|------|------|
| 1 | shuffle 操作是否完全一样？ | **是**，同一个 `validate()`，同样的 batch 内 `randperm`、同样的默认 batch_size 64。**但都是 batch 内打乱，不是全验证集打乱。** |
| 2 | 是否同一个 seed？ | **是**，都用默认 `ablation_seed=0`。**但同 seed ≠ 同置换**，见下。 |
| 3 | 是否同一种 evaluation？ | **是**。同一批 `(images, input_ids, labels)`，只换图片；同样的 per-token 加权平均 CE；`gap = shuffle − normal`。 |
| 4 | 还有哪些不一致？ | 有，见第 8 节：训练配置、验证集规模/抽样次数、配对泄漏率、loss 基线量级。 |

**关于第 2 点必须展开说明。** generator 是每次 `validate()` 调用新建、并在循环中被**连续消耗**的：
每个 batch 先抽 `randperm(B)`，再抽一个 `[B,3,64,64]` 的 noise 张量（消耗 `B×12288` 个随机数），
直接改变下一个 batch 的 `randperm` 状态。因此实际抽到的置换序列同时取决于
**batch 数量、每个 batch 的实际大小、image_size**：

- Flickr8k：4045 行 → 64 个 batch（63×64 + 13）；
- Toy：108 行 → **2 个 batch**（64 + 44）。

两边的置换实例完全不同。文书里可以写 "same fixed seed (0) for reproducibility"，
**不能**写 "identical shuffling" 或 "the same permutation"。

（另外 generator 建在 CUDA 上，换设备复现出的置换不同；loss 本身又叠加 bf16 autocast，
不同运行有末位抖动 —— 抖动幅度见 8.1 节实测。）

一致的部分（两边逐项核对过）：

| 项目 | Flickr8k | Toy | 一致 |
|------|----------|-----|------|
| 模型结构 | `build_model()`：TinyViT(patch 8) + TinyGPT(64d, 4h, 2L) | 同左 | ✅ |
| tokenizer | char-level，vocab 47 | char-level，vocab 47 | ✅ |
| image_size / batch_size | 64 / 64 | 64 / 64 | ✅ |
| 划分 | `split_by_image(0.1, seed=42)` | 同左 | ✅ |
| 精度 | CUDA + bf16 autocast | CUDA + bf16 autocast | ✅ |
| loss 定义 | per-token 平均 CE，`ignore_index=-100` | 同左 | ✅ |
| 模型状态 | `eval()` + `no_grad()`，加载 best checkpoint | 同左 | ✅ |

---

## 8. 已量化的四个不可比因素（写文书时要设的限）

### 8.1 数值噪声底与置换方差（有实测对照）

checkpoint 里保存了训练时最后一次 `validate()` 的结果，可以和后来独立脚本跑的 JSON 对照。
（注：从字段可以看出当时 `validate()` 还没有全部四路 —— Flickr8k checkpoint 只存了
`val_shuffle_loss`，toy checkpoint 多了 `val_zero_loss`，noise 路是后加的。
少一路 `torch.rand` 就少消耗 generator 状态，**因此两次跑抽到的是不同的置换**，
模型权重和验证集则完全相同。）

toy 侧因此一共留下了 **三次独立测量**（同一个模型、同一个验证集、同样的 `ablation_seed=0`）：

| toy 测量 | 来源 | 时间 | `val_loss` | `val_shuffle_loss` | **shuffle gap** | `val_zero_loss` |
|---|---|---|---|---|---|---|
| ① | `toy_visual_grounding.pt` 的 `validation` | 06-23 19:32 | 0.0368440 | 0.4186848 | **+0.3818** | 0.46473213 |
| ② | `logs/toy_grounding_metrics.json` | 06-23 19:33 | 0.0368440 | 0.3892166 | **+0.3524** | 0.46473213 |
| ③ | `logs/toy_grounding_ablation.json`（**现在引用的**） | 06-23 23:18 | 0.0369203 | 0.4024472 | **+0.3655** | 0.46466919 |

Flickr8k 侧有两次：

| flickr 测量 | 来源 | `val_loss` | `val_shuffle_loss` | **shuffle gap** |
|---|---|---|---|---|
| ① | `best_tinyvlm_shuffle.pt` 的 `validation` | 1.2984529 | 1.3124835 | **+0.0140** |
| ② | `logs/flickr8k_ablation.json`（**现在引用的**） | 1.2984940 | 1.3127709 | **+0.0143** |

从中可以直接读出两个量：

**(a) 数值噪声底 ≈ 1e-4。** `val_zero_loss` 的输入是确定的全黑图：①②（同一版代码）
**逐位相同** `0.46473212835251115`，说明同代码重复运行是确定性的；③ 差了 6.3e-5，
这是跨代码版本（多了一路 noise 前向，kernel 选择/规约顺序改变）带来的 bf16 抖动。
`val_loss` 同理，Flickr8k 两次差 4.1e-5。
→ Flickr8k 的 gap 0.0143 是噪声底的 ~140 倍，**小但真实，不是浮点误差**。
（同一抖动也让 toy 的 greedy 生成翻了 1/108 条预测：position accuracy 94.44% → 95.37%，
所以 `REVIEW_LOG.md:856` 引的是②的旧值，最终 JSON 是 95.37%。）

**(b) 置换本身的方差不可忽略，且两边差一个数量级。**
toy 三次的 gap 是 `0.3818 / 0.3524 / 0.3655`：mean ≈ 0.367，极差 0.029，
相当于 gap 的 **8%**——同一个模型、同一个 seed，只因为置换实例不同（toy 只有 **2 个 batch**、
即 2 次置换抽样）。Flickr8k 两次只差 0.0003（64 次抽样，平均效应强得多）。
→ toy 的 `0.3655` 不该按四位有效数字解读；两边都**没有跑多 seed、没有误差棒**。
如果要给 toy 一个诚实的数字，用这三次写成 `+0.37 ± 0.015` 比写 `+0.3655` 更站得住。

### 8.2 配对泄漏率不同（重要，且方向是低估）

因为是 batch 内随机置换、允许自映射，一部分"打乱后"的样本其实**没有被真正打乱**。
用 `ablation_seed=0` 精确重放置换后逐条统计：

| 泄漏类型 | Flickr8k | Toy |
|----------|----------|-----|
| 图片映射到自己（完全没变） | 70 / 4045 = **1.73%** | 1 / 108 = 0.93% |
| 打乱后仍是**同一张图**（Flickr8k 每图 5 条 caption 落在同一 batch） | 328 / 4045 = **8.11%** | 0.93% |
| 打乱后 caption **字面完全相同** | 1.73% | 5 / 108 = **4.63%** |

toy 还有属性层面的重合（只有 36 种 caption）：

```
打乱后仍同色 24.07%   仍同形状 36.11%   仍同位置 37.96%
```

两边的泄漏方向都是**让 gap 偏小**（把本该错配的样本算成了正确或部分正确配对），
所以两个数字都是保守估计。但**低估程度不同**：Flickr8k 的同图泄漏 8.11% 比 toy 的
0.93% 高近一个数量级 —— 横比时必须提一句。

### 8.3 训练配置不同：这不是控制变量实验

| 配置 | Flickr8k (`checkpoints/best_tinyvlm_shuffle.pt`) | Toy (`checkpoints/toy_visual_grounding.pt`) |
|------|---|---|
| epochs | 10 | 30 |
| lr | 3e-4 | 1e-3 |
| max_text_len | 96 | 32 |
| 验证集 | 4045 行 / 809 图 | 108 行 / 108 图 |
| 最终 val_loss | 1.2985（明显欠拟合） | 0.0369（几乎完全拟合） |

**两个 gap 来自两个不同的模型 × 两个不同的语料 × 两套超参**，只能作为"同一探针在两种难度下的读数"，
不能表述为同一实验的两个 condition。

### 8.4 绝对 gap 不能并排比较

| | val_loss | shuffle_loss | 绝对 gap | **相对 gap** | ppl（normal → shuffle） |
|---|---|---|---|---|---|
| Flickr8k | 1.2985 | 1.3128 | +0.0143 | **+1.10%** | 3.664 → 3.717 |
| Toy | 0.0369 | 0.4024 | +0.3655 | **+990%** | 1.038 → 1.496 |

两个数都是 nats/token，但一个是在 1.30 的基线上、另一个是在 0.037 的基线上，差两个数量级。
并排给出 `0.0143` 和 `0.3655` 而不给基线，是误导性的对比。

### 8.5 复现命令记录不完整

`README.md:85-93` 记了 Flickr8k 的 `run_image_ablation.py` 命令，
**没有记 toy 的 `evaluate_toy_grounding.py` 命令**。本文所有 toy 侧的 batch 结构和泄漏统计
都基于脚本默认值 `--batch_size 64 --num_workers 4`；若当时用了别的 batch_size，
8.1/8.2 的数字需按实际值重算（结论方向不变）。

---

## 9. 文书写法建议

推荐表述：

> To test whether the model actually conditions on the image rather than
> exploiting the language prior of the captions, we run an image-shuffle
> ablation: `input_ids` and `labels` are held fixed and only the image tensor is
> replaced by **another real image drawn from the same evaluation batch**
> (batch size 64, fixed seed 0), so the marginal image distribution is unchanged
> and only the image–text pairing is destroyed. We report the change in
> per-token validation cross-entropy. On Flickr8k the loss rises by 0.0143
> nats/token (1.2985 → 1.3128, +1.10%). On a synthetic shape dataset, whose
> captions are fully determined by three visual attributes and therefore carry
> almost no language prior, the same probe yields 0.3655 nats/token
> (0.0369 → 0.4024). We also report all-zero and uniform-noise images as
> auxiliary conditions, but treat shuffling as the primary evidence since the
> other two are out-of-distribution inputs.
>
> The two models are trained separately on different corpora and are not a
> controlled comparison. Because the permutation is drawn within each batch and
> is not a derangement, 8.1% of Flickr8k rows are still paired with a caption of
> the same image (0.9% for the synthetic set), so both gaps are conservative.

需要避免的说法：

- ❌ "images are shuffled across the validation set" —— 实际是 batch 内。
- ❌ "the image is replaced by noise" 用来描述 shuffle —— shuffle 用的是真实照片；噪声是另一条对照。
- ❌ "under the same intervention" 不加限定 —— 置换实例不同、泄漏率不同、模型不同。
- ❌ 只并排给绝对 gap 而不给 baseline —— 见 8.4。
- ❌ 给 toy 的 0.3655 赋予四位有效数字的精度 —— 见 8.1（三次测量摆动 8%）。
- ❌ 用 toy 的大 gap 反推「Flickr8k 上是视觉能力太弱」—— toy 同时换掉了数据难度、
  语言先验、训练时长和 caption 长度，四者混淆，见第 5 节。
- ❌ 把消融说成训练时的干预 —— 它只发生在 validation，训练全程都是正确配对，见 1.1。

---

## 10. 若要把它做成真正可比的实验（可选，不需要重训）

1. **全局 derangement**：把 batch 内 `randperm` 换成跨整个 val set 的置换，并显式排除
   自映射与同图映射 → 消除 8.2 的泄漏；
2. **多 seed**：`ablation_seed ∈ {0..9}` 各跑一次，报告 mean ± std → 解决 8.1 的置换方差；
3. **同时报告绝对与相对 gap**（8.4）；
4. **把 toy 的评估命令补进 `README.md`**（8.5）。

改动只涉及 `validate()` 里置换生成的那几行和调用脚本，模型权重不用动。

---

## 附：相关文件与数字索引

```text
tinyvlm/train_vlm.py:64-122        validate()：normal / shuffle / zero / noise 四路 loss
tinyvlm/train_vlm.py:87-93         置换生成（batch 内 randperm + 恒等兜底）
tinyvlm/train_vlm.py:22-43         build_model()：patch 8、d_model 64、2 层
tinyvlm/tinyvlm_model.py:99-122    visual prefix 拼接与 prefix mask
tinyvlm/tinyvlm_model.py:148-159   image 位置 label = -100、cross_entropy
tinyvlm/dataset.py:70-96           图片预处理（[0,1]、无 normalize）与数据侧 label shift
tinyvlm/dataset.py:124-147         split_by_image()：按图片划分
tinyvlm/toy_grounding.py           合成 grounding 数据集生成
scripts/run_image_ablation.py      Flickr8k 消融入口
scripts/evaluate_toy_grounding.py  Toy 消融 + 属性准确率入口
logs/flickr8k_ablation.json        val 1.298494 / shuffle 1.312771 / gap +0.014277
logs/toy_grounding_ablation.json   val 0.036920 / shuffle 0.402447 / gap +0.365527
REVIEW_LOG.md:783-847              当时的实验记录
```
