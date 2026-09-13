import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_block import TransformerBlock


class TinyGPT(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        num_layers,
        vocab_size,
        max_seq_len,
        tie_word_embeddings=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.tie_word_embeddings = tie_word_embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads) for _ in range(num_layers)]
        )
        self.final_ln = nn.LayerNorm(d_model)
        self.lm = nn.Linear(d_model, vocab_size)
        if tie_word_embeddings:
            # Embedding selects rows from [V,D]; Linear scores with hidden @ weight.T.
            # Preserve Linear's smaller output-projection initialization. The
            # default Embedding N(0,1) scale produces excessively large tied logits.
            with torch.no_grad():
                self.token_embedding.weight.copy_(self.lm.weight)
            self.lm.weight = self.token_embedding.weight

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).bool()
        self.register_buffer(
            "casual_mask", mask.view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(self, inputs_id, targets=None, past_kvs=None, use_cache=False):
        _, seq_len = inputs_id.shape
        past_len = past_kvs[0][0].shape[2] if past_kvs is not None else 0
        total_len = past_len + seq_len
        if total_len > self.max_seq_len:
            raise ValueError(
                f"total sequence length {total_len} exceeds max_seq_len {self.max_seq_len}"
            )

        position_ids = torch.arange(past_len, total_len, device=inputs_id.device)
        x = self.token_embedding(inputs_id)
        x = x + self.position_embedding(position_ids)
        mask = self.casual_mask[:, :, past_len:total_len, :total_len]

        present_kvs = [] if use_cache else None
        for index, block in enumerate(self.blocks):
            past_kv = past_kvs[index] if past_kvs is not None else None
            if use_cache:
                x, present_kv = block(x, mask, past_kv, use_cache=True)
                present_kvs.append(present_kv)
            else:
                x = block(x, mask)

        logits = self.lm(self.final_ln(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size), targets.reshape(-1)
            )
        if use_cache:
            return logits, loss, present_kvs
        return logits, loss
