"""
evolution.py
-------------
策略演化模块，包括Fermi更新规则。
"""

import math
import random


def fermi_update(defender_i, defender_j, K=0.1):
    """
    Fermi策略更新规则
    ------------------
    防御者i根据邻居j的收益决定是否模仿其策略。

    Args:
        defender_i (Defender): 被更新的防御者
        defender_j (Defender): 参考的邻居防御者
        K (float): 演化温度参数（越大越随机）
    Returns:
        None（直接更新defender_i的策略）
    """
    payoff_diff = defender_j.payoff - defender_i.payoff

    # 防止数学溢出，限制指数函数的输入范围
    if payoff_diff / K > 100:
        prob = 1.0
    elif payoff_diff / K < -100:
        prob = 0.0
    else:
        prob = 1 / (1 + math.exp(-payoff_diff / K))

    if random.random() < prob:
        defender_i.strategy = defender_j.strategy
