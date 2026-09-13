import torch
import torch.nn as nn
import torch.nn.functional as F


def build_prefix_mask(image_length, text_length, device=None):
    """Build a visual-prefix/text-causal attention mask.

    Visual queries attend bidirectionally within the visual prefix but cannot
    read text. Text queries attend to every visual token and causal text.
    """
    total_length = image_length + text_length
    mask = torch.zeros(
        total_length, total_length, dtype=torch.bool, device=device
    )
    mask[:image_length, :image_length] = True
    mask[image_length:, :image_length] = True
    mask[image_length:, image_length:] = torch.tril(
        torch.ones(text_length, text_length, dtype=torch.bool, device=device)
    )
    return mask.view(1, 1, total_length, total_length)


def build_decode_mask(past_length, text_length, device=None):
    """Let new text queries see the cache and causal positions within the chunk."""
    mask = torch.ones(
        text_length,
        past_length + text_length,
        dtype=torch.bool,
        device=device,
    )
    mask[:, past_length:] = torch.tril(
        torch.ones(text_length, text_length, dtype=torch.bool, device=device)
    )
    return mask.view(1, 1, text_length, past_length + text_length)


class TinyVLM(nn.Module):
    """Bridge visual tokens from TinyViT into the TinyGPT token space."""

    def __init__(self, vit, gpt, vit_dim, gpt_dim):
        super().__init__()
        self.vit = vit
        self.gpt = gpt
        self.visual_proj = nn.Linear(vit_dim, gpt_dim)

    def _vocab_projection(self, hidden_states):
        # TinyGPT versions in this teaching project may call this layer lm or lm_head.
        output_layer = getattr(self.gpt, "lm_head", None)
        if output_layer is None:
            output_layer = getattr(self.gpt, "lm", None)
        if output_layer is None:
            raise AttributeError("TinyGPT must define either 'lm' or 'lm_head'")
        return output_layer(hidden_states)

    def forward(
        self,
        images,
        input_ids,
        labels=None,
        past_kvs=None,
        use_cache=False,
    ):
        """
        images:    [B,3,H,W]
        input_ids: [B,T_text]
        labels:    [B,T_text], with text padding already set to -100
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B,T_text]")
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels and input_ids must have the same shape")

        batch_size, text_length = input_ids.shape
        if past_kvs is not None:
            if images is not None:
                raise ValueError("images must be None during cached decode")
            if not use_cache:
                raise ValueError("past_kvs requires use_cache=True")
            if labels is not None:
                raise ValueError("labels are not supported during cached decode")
            if len(past_kvs) != len(self.gpt.blocks):
                raise ValueError("past_kvs must contain one entry per GPT block")

            past_length = past_kvs[0][0].size(2)
            total_length = past_length + text_length
            position_ids = torch.arange(
                past_length, total_length, device=input_ids.device
            )
            hidden_states = self.gpt.token_embedding(input_ids)
            hidden_states = hidden_states + self.gpt.position_embedding(position_ids)
            mask = build_decode_mask(
                past_length, text_length, device=hidden_states.device
            )
            image_length = 0
        else:
            if images is None:
                raise ValueError("images are required during multimodal prefill")
            image_tokens = self.vit(images)
            if image_tokens.ndim != 3 or image_tokens.size(0) != batch_size:
                raise AssertionError(
                    "TinyViT must return image tokens with shape [B,N_img,vit_dim]"
                )
            if image_tokens.size(-1) != self.visual_proj.in_features:
                raise AssertionError(
                    "TinyViT output dimension does not match visual_proj input dimension"
                )

            image_embeddings = self.visual_proj(image_tokens)  # [B,N_img,gpt_dim]
            text_embeddings = self.gpt.token_embedding(input_ids)  # [B,T_text,gpt_dim]
            hidden_states = torch.cat(
                [image_embeddings, text_embeddings], dim=1
            )  # [B,N_img+T_text,gpt_dim]
            total_length = hidden_states.size(1)
            position_ids = torch.arange(total_length, device=hidden_states.device)
            hidden_states = hidden_states + self.gpt.position_embedding(position_ids)
            image_length = image_embeddings.size(1)
            mask = build_prefix_mask(
                image_length,
                text_length,
                device=hidden_states.device,
            )

        if total_length > self.gpt.max_seq_len:
            raise ValueError(
                f"multimodal sequence length {total_length} exceeds "
                f"TinyGPT max_seq_len {self.gpt.max_seq_len}"
            )

        present_kvs = [] if use_cache else None
        for block_index, block in enumerate(self.gpt.blocks):
            past_kv = past_kvs[block_index] if past_kvs is not None else None
            if use_cache:
                hidden_states, present_kv = block(
                    hidden_states,
                    mask,
                    past_kv=past_kv,
                    use_cache=True,
                )
                present_kvs.append(present_kv)
            else:
                hidden_states = block(hidden_states, mask)
        hidden_states = self.gpt.final_ln(hidden_states)
        logits = self._vocab_projection(hidden_states)  # [B,N_img+T_text,V]

        loss = None
        if labels is not None:
            image_labels = torch.full(
                (batch_size, image_length),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            full_labels = torch.cat([image_labels, labels], dim=1)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                full_labels.reshape(-1),
                ignore_index=-100,
            )

        if use_cache:
            return logits, loss, present_kvs
        return logits, loss
