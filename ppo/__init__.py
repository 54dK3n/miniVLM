"""多模态 PPO（RLHF）模块。

工程骨架（数据 / rollout / value head / reward 接口 / 训练循环）已搭好；
四个 RL 数学核心留在 ppo/ppo_core.py 里作 NotImplementedError，由你手写：
    compute_rewards / compute_gae / ppo_policy_loss / ppo_value_loss
实现它们之后 train_ppo.py 即可端到端跑。
"""
