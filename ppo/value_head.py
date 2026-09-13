"""ActorCritic：给 TinyVLM 挂一个 value head（critic），一次前向同时拿 logits 和 value。

不改动 TinyVLM 本体：用 forward hook 抓 gpt.final_ln 的输出（最后一层 hidden，
形状 [B, N_img+T, D]），再过一个 Linear(D,1) 得到逐位置 value [B, N_img+T]。

PPO 要优化的参数 = policy 本体 + value_head；reference 另用 clone_frozen 复制冻结。
"""

import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, model, gpt_dim=None):
        super().__init__()
        self.model = model  # TinyVLM
        if gpt_dim is None:
            gpt_dim = model.gpt.token_embedding.embedding_dim
        self.value_head = nn.Linear(gpt_dim, 1)

        # hook 抓 final_ln 输出（prefill 时 = [B, N_img+T, D]）
        self._hidden = None
        model.gpt.final_ln.register_forward_hook(self._capture_hidden)

    def _capture_hidden(self, module, inputs, output):
        self._hidden = output

    def forward(self, images, input_ids):
        """返回 (logits [B, N_img+T, V], values [B, N_img+T])。"""
        logits, _ = self.model(images, input_ids)
        values = self.value_head(self._hidden).squeeze(-1)  # [B, N_img+T]
        self._hidden = None
        return logits, values

    @property
    def num_image_tokens_hint(self):
        """N_img 一般从 logits/text 形状反推（见 ppo_core._text_slice），这里不硬编码。"""
        return None
