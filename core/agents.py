"""
agents.py
定义系统中的智能体类，包括：
- 防御者（Defender）
- 攻击者（Attacker）
"""

import random


class Agent:
    """
    基础智能体类（抽象基类）
    -------------------------
    用于定义所有智能体的共同属性，如ID、策略、收益。
    """

    def __init__(self, agent_id, strategy='C'):
        """
        Args:
            agent_id (int): 智能体编号
            strategy (str): 当前策略，'C'表示合作(Cooperate)，'D'表示背叛(Defect)
        """
        self.id = agent_id
        self.strategy = strategy
        self.payoff = 0.0  # 当前累计收益

    def reset_payoff(self):
        """
        将智能体的收益重置为0（每轮博弈后调用）
        Returns:
            None
        """
        self.payoff = 0.0


class Defender(Agent):
    """
    防御者类（继承自Agent）
    ---------------------
    表示网络中的一个防御节点，其策略为 C（投资防御/合作）或 D（不投资/背叛）。
    """

    def __init__(self, agent_id, strategy='C'):
        """
        Args:
            agent_id (int): 节点编号
            strategy (str): 初始策略（'C' 或 'D'）
        """
        super().__init__(agent_id, strategy)
        self.neighbors = []  # 存储相邻防御节点的列表


class Attacker:
    """
    攻击者类
    --------
    表示对网络进行攻击的主体，具有攻击概率q和自适应反馈机制。
    """

    def __init__(self, q0=0.5, alpha=0.0, cost=10, reward=50):
        """
        Args:
            q0 (float): 初始攻击概率 (0~1)
            alpha (float): 攻击反馈速率 αA，用于更新攻击概率
            cost (float): 攻击成本
            reward (float): 攻击成功后的收益
        """
        self.q = q0
        self.alpha = alpha
        self.cost = cost
        self.reward = reward

    def should_attack(self):
        """
        根据当前攻击概率q决定是否发起攻击。
        Returns:
            bool: True 表示发动攻击，False 表示不攻击
        """
        return random.random() < self.q

    def update_feedback(self, local_success, global_success):
        """
        根据攻击成功率更新攻击概率 q。
        对应论文公式：q_i(t+1) = q_i(t) + αA * (τ_i - τ)

        Args:
            local_success (float): 本地攻击成功率（该轮攻击成功次数 / 总攻击次数）
            global_success (float): 全局平均攻击成功率
        Returns:
            None
        """
        self.q += self.alpha * (local_success - global_success)
        self.q = max(0, min(1, self.q))  # 限制在[0,1]区间
