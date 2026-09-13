"""Reward 接口：给 (image, prompt, response_text) 打一个标量分。

PPO 的 reward 来源是可插拔的——真实 RLHF 用一个训练好的 reward model，本教学项目
先给两个占位实现，保证 pipeline 能跑通；之后替换成 CLIP 分 / 训练好的 RM 即可，
train_ppo.py 里只依赖 `score_batch(images, prompts, responses) -> [B] tensor`。

约定：分数越高越好；PPO 会最大化它（并被 KL 惩罚拉回 reference）。
"""

import torch


class RewardModel:
    """打分器协议。实现 score_batch 即可接入 PPO。"""

    def score_batch(self, images, prompts, responses):
        """images: [B,3,H,W]；prompts/responses: list[str]；返回 [B] float tensor。"""
        raise NotImplementedError


class LengthReward(RewardModel):
    """占位奖励：鼓励非空、适度长度、含 EOS 的回复（仅用于打通链路，非真实信号）。

    target_len 附近给高分，过短/过长扣分。真实使用时请替换成 CLIP/RM。
    """

    def __init__(self, target_len=40, scale=0.02):
        self.target_len = target_len
        self.scale = scale

    def score_batch(self, images, prompts, responses):
        scores = []
        for text in responses:
            length = len(text.strip())
            scores.append(-self.scale * abs(length - self.target_len))
        return torch.tensor(scores, dtype=torch.float32, device=images.device)


class ConstantReward(RewardModel):
    """全 0 奖励：验证 pipeline（此时 PPO 只受 KL 惩罚驱动，advantage≈0）。"""

    def score_batch(self, images, prompts, responses):
        return torch.zeros(images.size(0), dtype=torch.float32, device=images.device)
