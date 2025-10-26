"""
games.py
---------
实现两种博弈模型：
1. 公共物品博弈（Public Goods Game, PGG）
2. 攻防博弈（Defender-Attacker Game）
"""

import random


class PublicGoodsGame:
    """
    公共物品博弈类
    ---------------
    每个参与者可选择投资或不投资，投资者付出成本但共享收益。
    """

    def __init__(self, r, mu=40):
        """
        Args:
            r (float): 增强因子（放大公共收益）
            mu (float): 每个合作者的防御投资成本 μ
        """
        self.r = r
        self.mu = mu

    def play(self, group):
        """
        在一组防御者之间进行公共物品博弈。
        Args:
            group (list[Defender]): 参与博弈的一组防御者
        Returns:
            None （直接更新各个防御者的payoff）
        """
        contributors = [a for a in group if a.strategy == 'C']
        total = len(contributors) * self.mu
        benefit = self.r * total / len(group)
        for a in group:
            if a.strategy == 'C':
                a.payoff += benefit - self.mu
            else:
                a.payoff += benefit


class DefenderAttackerGame:
    """
    攻防博弈类
    ------------
    模拟单个防御者与攻击者之间的收益交互。
    防御者的投资能力由其焦点组的集体贡献决定。
    """

    def __init__(self, gamma1=50, gamma2=10, delta=50, d=50, c=10):
        """
        Args:
            gamma1 (float): 投资且被攻时的收益 γ1
            gamma2 (float): 投资且未被攻时的收益 γ2
            delta (float): 未投资且被攻时的损失 δ
            d (float): 攻击者成功攻击的收益
            c (float): 攻击者的攻击成本
        """
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.delta = delta
        self.d = d
        self.c = c

    def play(self, defender, attacker, focal_group=None):
        """
        计算单次攻防博弈的收益。

        Args:
            defender (Defender): 防御者对象
            attacker (Attacker): 攻击者对象
            focal_group (list[Defender]): 防御者所在的组（包含自身和邻居）
        Returns:
            tuple(float, float): (防御者收益, 攻击者收益)
        """
        attack = attacker.should_attack()

        # 根据防御者所在的组的集体贡献决定防御者的投资概率
        if focal_group is None:
            # 如果没有提供防御者所在的组，使用个体策略（向后兼容）
            invest = defender.strategy == 'C'
        else:
            # 计算防御者所在的组中的合作者数量
            contributors = [d for d in focal_group if d.strategy == 'C']
            investment_probability = len(contributors) / len(focal_group)

            # 根据概率决定是否投资
            invest = random.random() < investment_probability

        # 集体投资的回报和支出这一项在PublicGoodsGame类中单独计算，并在每次博弈后直接更新
        # 投资且被攻击
        if invest and attack:
            return self.gamma1, -self.c
        # 投资且未被攻击
        elif invest and not attack:
            return self.gamma2, 0
        # 未投资且被攻击
        elif not invest and attack:
            return -self.delta, self.d
        # 未投资且未被攻击
        else:
            return 0, 0
