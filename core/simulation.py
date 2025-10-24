"""
simulation.py
--------------
整合所有模块（拓扑、博弈、演化）实现完整的仿真流程。
"""

import random
from core.agents import Defender, Attacker
from core.games import PublicGoodsGame, DefenderAttackerGame
from core.topology import NetworkTopology
from core.evolution import fermi_update


class CyberSecuritySimulation:
    """
    网络安全博弈仿真主类
    -------------------
    负责初始化网络、运行每轮博弈、更新策略与记录数据。
    """

    def __init__(self, N=1600, rounds=2000, r=4.0, q0=0.4, alphaA=0.0, K=0.1, topology='lattice', params=None):
        """
        Args:
            N (int): 节点数
            rounds (int): 仿真轮数
            r (float): 公共物品博弈增强系数
            q0 (float): 初始攻击概率
            alphaA (float): 攻击者反馈速率
            K (float): Fermi更新温度
            topology (str): 网络类型
            params (dict): 拓扑参数
        """
        self.network = NetworkTopology(topology, N, params)
        self.defenders = [Defender(i, random.choice(['C', 'D'])) for i in range(N)]
        self.attacker = Attacker(q0=q0, alpha=alphaA)
        self.pgg = PublicGoodsGame(r)
        self.dag = DefenderAttackerGame()
        self.rounds = rounds
        self.K = K

    def run(self, recorder):
        """
        执行完整仿真过程。

        Args:
            recorder (DataRecorder): 数据记录对象
        Returns:
            None（结果通过recorder记录）
        """
        for t in range(self.rounds):
            # === 1. 公共物品博弈 ===
            for node in self.network.graph.nodes:
                neighbors = list(self.network.graph.neighbors(node))
                group = [self.defenders[node]] + [self.defenders[n] for n in neighbors]
                self.pgg.play(group)

            # === 2. 攻防博弈 ===
            attack_success, total_attacks = 0, 0
            for d in self.defenders:
                dp, ap = self.dag.play(d, self.attacker)
                d.payoff += dp
                if dp < 0:  # 攻击成功的判定
                    attack_success += 1
                total_attacks += 1

            # === 3. 攻击者反馈更新 ===
            local_succ = attack_success / total_attacks
            self.attacker.update_feedback(local_succ, local_succ)

            # === 4. 防御者策略更新 ===
            for d in self.defenders:
                neighbor_id = random.choice(list(self.network.graph.neighbors(d.id)))
                fermi_update(d, self.defenders[neighbor_id], self.K)

            # === 5. 记录数据 ===
            coop_rate = sum(d.strategy == 'C' for d in self.defenders) / len(self.defenders)
            recorder.record(coop_rate, local_succ, self.attacker.q)

            # === 6. 重置收益 ===
            for d in self.defenders:
                d.reset_payoff()
