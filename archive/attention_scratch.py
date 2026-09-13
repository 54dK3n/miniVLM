import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import time

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        assert d_model%num_heads==0
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=d_model//num_heads
        self.q_projection=nn.Linear(d_model,d_model)
        self.k_projection=nn.Linear(d_model,d_model)
        self.v_projection=nn.Linear(d_model,d_model)
        self.out_projection=nn.Linear(d_model,d_model)
    def forward(self,x,mask=None,past_kv=None,use_cache=False):
        B,T,D=x.shape
        q=self.q_projection(x)
        k=self.k_projection(x)
        v=self.v_projection(x)
        q=q.view(B,T,self.num_heads,self.d_k).transpose(1,2)
        k=k.view(B,T,self.num_heads,self.d_k).transpose(1,2)
        v=v.view(B,T,self.num_heads,self.d_k).transpose(1,2)
        if past_kv is not None:
            past_k,past_v=past_kv
            k=torch.cat([past_k,k],dim=2)
            v=torch.cat([past_v,v],dim=2)
        past_kv=(k,v) if use_cache else None
        output=self.scaled_dot_attention(q,k,v,mask)
        output=output.transpose(1,2).contiguous().view(B,T,D)
        output=self.out_projection(output)
        if use_cache:
            return output,past_kv
        return output
    def scaled_dot_attention(self,q, k, v, mask=None):
        mid_output = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        if mask is not None:
            mid_output=mid_output.masked_fill(mask ==0,float('-inf'))
        attention_weights = F.softmax(mid_output, dim=-1)
        return attention_weights @ v
    def mask_padding(self,T):
        mask=torch.tril(torch.ones(T,T)).bool()
        return mask.view(1,1,T,T)
class FeedForward(nn.Module):
    def __init__(self,d_model,hidden_dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d_model,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,d_model))
    def forward(self,x):
        return self.net(x)
class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads,ratio=4):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.hidden_dim=d_model*ratio
        self.ln1=nn.LayerNorm(d_model)
        self.attn=MultiHeadAttention(d_model,num_heads)
        self.ln2=nn.LayerNorm(d_model)
        self.mlp=FeedForward(d_model,self.hidden_dim)
    def forward(self,x,mask=None,past_kv=None,use_cache=False):
        residual=x
        x1=self.ln1(x)
        if use_cache:
            attn_output, present_kv = self.attn(
                x1,
                mask=mask,
                past_kv=past_kv,
                use_cache=True
            )
        else:
            attn_output = self.attn(
                x1,
                mask=mask,
                past_kv=None,
                use_cache=False
            )
            present_kv = None
        x1=attn_output+residual
        x2=self.ln2(x1)
        mlp_output=self.mlp(x2)
        if use_cache:
            return mlp_output+x1, present_kv
        return mlp_output+x1
class TinyGPT(nn.Module):
    def __init__(self,d_model,num_heads,num_layers,vocab_size,max_seq_len):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.num_layers=num_layers
        self.vocab_size=vocab_size
        self.max_seq_len=max_seq_len
        self.token_embedding=nn.Embedding(vocab_size,d_model)
        self.position_embedding=nn.Embedding(max_seq_len,d_model)

        self.blocks=nn.ModuleList([TransformerBlock(d_model,num_heads) for _ in range(num_layers)])
        self.final_ln=nn.LayerNorm(d_model)
        self.lm=nn.Linear(d_model,vocab_size)

        mask=torch.tril(torch.ones(max_seq_len,max_seq_len)).bool().view(1,1,max_seq_len, max_seq_len)
        self.register_buffer("casual_mask",mask)
    def forward(self,inputs_id,targets=None,past_kvs=None, use_cache=False):
        B,T=inputs_id.shape
        if past_kvs is not None:
            past_len = past_kvs[0][0].shape[2]
        else:
            past_len = 0
        total_len = past_len + T

        if total_len > self.max_seq_len:
            raise ValueError(
                f"total sequence length {total_len} exceeds max_seq_len {self.max_seq_len}"
            )
        token_embedding=self.token_embedding(inputs_id)
        position_ids=torch.arange(past_len,total_len,device=inputs_id.device)
        position_embedding=self.position_embedding(position_ids)
        x=token_embedding+position_embedding

        mask=self.casual_mask[:, :,past_len:total_len, :total_len]
        present_kvs = [] if use_cache else None
        for i,block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            if  use_cache:
                x,present_kv=block(
                    x,mask=mask,past_kv=past_kv,use_cache=True
                )
                present_kvs.append(present_kv)
            else:
                x=block(x,mask,past_kv=None,use_cache=False)
        x=self.final_ln(x)
        logits=self.lm(x)
        loss=None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1)
            )
        if use_cache:
            return logits, loss, present_kvs
        return logits, loss

def get_batch(data,batch_size,block_size,device='cpu'):
     ix=torch.randint(0,len(data)-block_size-1,(batch_size,))
     x=torch.stack([data[i:i+block_size] for i in ix]).to(device)
     y=torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
     return x,y

@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size=32, eval_iters=20, device='cpu'):
    model.eval()
    out = {}

    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            x, y = get_batch(data, batch_size=batch_size, block_size=block_size, device=device)
            logits, loss = model(x, y)
            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()
    return out

def training_loop(model, train_data, val_data, epochs, batch_size=32,block_size=8,lr=1e-3,eval_interval=100,device='cpu'):
    model=model.to(device)
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    history = []
    for epoch in range(epochs):
        x,y=get_batch(train_data,batch_size,block_size,device)
        logits,loss=model(x,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch%eval_interval==0:
            losses=estimate_loss(model,train_data,val_data,block_size,device=device)
            train_loss = float(losses["train"])
            val_loss = float(losses["val"])

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss
            })

            print(
                f"Epoch {epoch}, "
                f"train loss: {train_loss:.4f}, "
                f"val loss: {val_loss:.4f}"
            )

    return history
    
@torch.no_grad()
def evaluate_model(
    model,
    train_data,
    val_data,
    block_size,
    batch_size=32,
    eval_iters=50,
    device="cpu"
):
    losses = estimate_loss(
        model=model,
        train_data=train_data,
        val_data=val_data,
        block_size=block_size,
        batch_size=batch_size,
        eval_iters=eval_iters,
        device=device
    )

    train_loss = float(losses["train"])
    val_loss = float(losses["val"])

    train_ppl = compute_complexity(train_loss)
    val_ppl = compute_complexity(val_loss)

    print("\n=== Evaluation ===")
    print(f"train loss: {train_loss:.4f}")
    print(f"val loss:   {val_loss:.4f}")
    print(f"train ppl:  {train_ppl:.4f}")
    print(f"val ppl:    {val_ppl:.4f}")

    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_ppl": train_ppl,
        "val_ppl": val_ppl,
    }
@torch.no_grad()
def benchmark_generate(
    model,
    start,
    max_new_tokens,
    block_size,
    temperature=1.0,
    top_k=None,
    device="cpu"
):
    model.eval()
    start = start.to(device)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    t0 = time.time()

    out = generate(
        model=model,
        idx=start,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        temperature=temperature,
        top_k=top_k
    )

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    t1 = time.time()

    total = t1 - t0
    tokens_per_sec = max_new_tokens / total

    print("\n=== Generate Benchmark: No KV Cache ===")
    print(f"max_new_tokens: {max_new_tokens}")
    print(f"time: {total:.4f} sec")
    print(f"tokens/sec: {tokens_per_sec:.2f}")

    return {
        "output": out,
        "time": total,
        "tokens_per_sec": tokens_per_sec,
    }

@torch.no_grad()
def sample_prompts(
    model,
    encode,
    decode,
    prompts,
    block_size,
    max_new_tokens=100,
    temperature=0.8,
    top_k=5,
    device="cpu"
):
    model.eval()

    print("\n=== Sample Generation ===")

    for prompt in prompts:
        idx = encode(prompt).unsqueeze(0).to(device)

        out = generate(
            model=model,
            idx=idx,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            temperature=temperature,
            top_k=top_k
        )

        text = decode(out[0])

        print("-" * 50)
        print(f"Prompt: {repr(prompt)}")
        print(text)

@torch.no_grad()
def generate(
    model,
    idx,
    max_new_tokens,
    block_size,
    temperature=1.0,
    top_k=None,
    use_kv_cache=True
):
    model.eval()

    # 只取最后 block_size 个 token 作为初始上下文
    idx_cond = idx[:, -block_size:]

    # KV cache 版本要求：初始上下文长度 + 新生成长度 <= max_seq_len
    # 否则 position embedding 会超出范围
    can_use_cache = (
        use_kv_cache
        and idx_cond.size(1) + max_new_tokens <= model.max_seq_len
    )

    if can_use_cache:
        # =========================
        # KV cache generation
        # =========================

        # 第一次：prefill，把 prompt 整体喂进去，建立 KV cache
        logits, loss, past_kvs = model(
            idx_cond,
            past_kvs=None,
            use_cache=True
        )

        for step in range(max_new_tokens):
            # 当前 logits 的最后一个位置，用来预测下一个 token
            logits_last = logits[:, -1, :] / temperature

            if top_k is not None:
                k = min(top_k, logits_last.size(-1))
                values, indices = torch.topk(logits_last, k)
                filtered_logits = torch.full_like(logits_last, float("-inf"))
                filtered_logits.scatter_(1, indices, values)
                logits_last = filtered_logits

            probs = F.softmax(logits_last, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_token), dim=1)

            # 最后一步不需要再算下一轮 logits
            if step == max_new_tokens - 1:
                break

            # 关键：之后每次只喂新生成的 1 个 token
            logits, loss, past_kvs = model(
                next_token,
                past_kvs=past_kvs,
                use_cache=True
            )

        return idx

    else:
        # =========================
        # Original no-cache generation
        # =========================
        # 如果超过 max_seq_len，就退回你原来的逻辑：
        # 每次重新取最后 block_size 个 token 计算

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]

            logits, loss = model(idx_cond)

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                values, indices = torch.topk(logits, k)
                filtered_logits = torch.full_like(logits, float("-inf"))
                filtered_logits.scatter_(1, indices, values)
                logits = filtered_logits

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_token), dim=1)

        return idx
def tokenizer(text):
    chars=sorted(list(set(text)))
    stoi={ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for ch,i in stoi.items()}
    def encode(s):
        return torch.tensor([stoi[c] for c in s], dtype=torch.long)
    def decode(ids):
        return ''.join(itos[int(i)] for i in ids)
    return encode, decode,len(chars),stoi, itos

def save_checkpoint(
    path,
    model,
    vocab_size,
    max_seq_len,
    d_model,
    num_heads,
    num_layers,
    stoi,
    itos,
):
    ckpt={
        "model_state_dict":model.state_dict(),
        "config":{
            "vocab_size":vocab_size,
            "max_seq_len":max_seq_len,
            "d_model":d_model,
            "num_heads":num_heads,
            "num_layers":num_layers,
        },
        "stoi": stoi,
        "itos": itos,
    }
    torch.save(ckpt,path)

def compute_complexity(loss):
    return math.exp(loss)

def test_mha_kv_cache():
    torch.manual_seed(42)

    mha = MultiHeadAttention(d_model=16, num_heads=4)

    x1 = torch.randn(2, 3, 16)
    x2 = torch.randn(2, 1, 16)

    out1, kv1 = mha(x1, use_cache=True)

    out2, kv2 = mha(x2, past_kv=kv1, use_cache=True)

    print("out1 shape:", out1.shape)
    print("kv1 k shape:", kv1[0].shape)
    print("kv1 v shape:", kv1[1].shape)

    print("out2 shape:", out2.shape)
    print("kv2 k shape:", kv2[0].shape)
    print("kv2 v shape:", kv2[1].shape)

def test_block_kv_cache():
    torch.manual_seed(42)

    block = TransformerBlock(d_model=16, num_heads=4)

    x1 = torch.randn(2, 3, 16)
    x2 = torch.randn(2, 1, 16)

    out1, kv1 = block(x1, use_cache=True)
    out2, kv2 = block(x2, past_kv=kv1, use_cache=True)

    print("out1 shape:", out1.shape)
    print("kv1 k shape:", kv1[0].shape)
    print("kv1 v shape:", kv1[1].shape)

    print("out2 shape:", out2.shape)
    print("kv2 k shape:", kv2[0].shape)
    print("kv2 v shape:", kv2[1].shape)
def test_tinygpt_kv_cache():
    torch.manual_seed(42)

    model = TinyGPT(
        d_model=16,
        num_heads=4,
        num_layers=2,
        vocab_size=20,
        max_seq_len=8
    )

    idx1 = torch.randint(0, 20, (2, 3))
    idx2 = torch.randint(0, 20, (2, 1))

    logits1, loss1, kvs1 = model(idx1, use_cache=True)
    logits2, loss2, kvs2 = model(idx2, past_kvs=kvs1, use_cache=True)

    print("logits1 shape:", logits1.shape)
    print("logits2 shape:", logits2.shape)

    print("num layers in kvs1:", len(kvs1))
    print("num layers in kvs2:", len(kvs2))

    for i in range(len(kvs1)):
        print(f"layer {i} kvs1 k shape:", kvs1[i][0].shape)
        print(f"layer {i} kvs1 v shape:", kvs1[i][1].shape)

        print(f"layer {i} kvs2 k shape:", kvs2[i][0].shape)
        print(f"layer {i} kvs2 v shape:", kvs2[i][1].shape)


if __name__ == "__main__":
    test_tinygpt_kv_cache()
    exit()
    torch.manual_seed(42)

    text = """
    hello tiny gpt.
    this is a small character language model.
    hello hello hello.
    tiny gpt learns next character prediction.
    """ * 200

    encode, decode, vocab_size, stoi, itos = tokenizer(text)

    data = encode(text)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    max_seq_len = 64
    block_size = 64

    model = TinyGPT(
    d_model=64,
    num_heads=4,
    num_layers=2,
    vocab_size=vocab_size,
    max_seq_len=max_seq_len
)

    history = training_loop(
    model=model,
    train_data=train_data,
    val_data=val_data,
    epochs=1000,
    batch_size=32,
    block_size=block_size,
    lr=1e-3,
    eval_interval=100,
    device="cpu"
)
    metrics = evaluate_model(
    model=model,
    train_data=train_data,
    val_data=val_data,
    block_size=block_size,
    batch_size=32,
    eval_iters=50,
    device="cpu"
)
    save_checkpoint(
    "tinygpt_char.pt",
    model,
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    d_model=64,
    num_heads=4,
    num_layers=2,
    stoi=stoi,
    itos=itos,
 )
    sample_prompts(
    model=model,
    encode=encode,
    decode=decode,
    prompts=[
        "hello",
        "tiny",
        "this ",
        "gpt "
    ],
    block_size=block_size,
    max_new_tokens=100,
    temperature=0.8,
    top_k=5,
    device="cpu"
)   
    start = encode("hello").unsqueeze(0)

    benchmark_result = benchmark_generate(
        model=model,
        start=start,
        max_new_tokens=200,
        block_size=block_size,
        temperature=0.8,
        top_k=5,
        device="cpu"
    )
    start = encode("hello").unsqueeze(0)

    generated = generate(
        model,
        start,
        max_new_tokens=100,
        block_size=block_size,
        temperature=0.8
    )

    print(decode(generated[0]))
