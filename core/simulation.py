"""
simulation.py
--------------
整合所有模块（拓扑、博弈、演化）实现完整的仿真流程。
提供抽象基类用于不同实验的仿真。
"""

import random
from abc import ABC, abstractmethod
from core.agents import Defender, Attacker
from core.games import PublicGoodsGame, DefenderAttackerGame
from core.topology import NetworkTopology
from core.evolution import fermi_update


class CyberSecuritySimulation(ABC):
    """
    网络安全博弈仿真抽象基类
    -------------------
    提供通用的仿真框架，具体实验需要继承并实现抽象方法。
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
        self.N = N
        self.rounds = rounds
        self.r = r
        self.q0 = q0
        self.alphaA = alphaA
        self.K = K
        self.topology = topology
        self.params = params

        # 初始化组件
        self.network = NetworkTopology(topology, N, params)
        self.defenders = [Defender(i, random.choice(['C', 'D'])) for i in range(N)]
        self.attacker = Attacker(q0=q0, alpha=alphaA)
        self.pgg = PublicGoodsGame(r)
        self.dag = DefenderAttackerGame()

        # 为每个防御者设置邻居节点
        for defender in self.defenders:
            if topology == 'lattice':
                # 对于2D格子网络，需要将防御者的整数ID转换为坐标，然后获取邻居
                L = int(N ** 0.5)
                coord = (defender.id // L, defender.id % L)
                neighbor_coords = list(self.network.graph.neighbors(coord))
                # 将邻居坐标转换回防御者索引
                defender.neighbors = [self.defenders[n[0] * L + n[1]] for n in neighbor_coords]
            else:
                # 对于其他网络拓扑，直接使用整数ID
                defender.neighbors = [self.defenders[neighbor_id] for neighbor_id in self.network.graph.neighbors(defender.id)]

    @abstractmethod
    def run(self, recorder):
        """
        执行仿真过程 - 子类必须实现此方法

        Args:
            recorder (DataRecorder): 数据记录对象
        Returns:
            None（结果通过recorder记录）
        """
        pass

    def run_standard_simulation(self, recorder):
        """
        标准仿真流程 - 默认的仿真实现
        子类可以直接调用此方法实现标准仿真

        Args:
            recorder (DataRecorder): 数据记录对象
        """
        for t in range(self.rounds):
            # === 1. 公共物品博弈 ===
            for node in self.network.graph.nodes:
                neighbors = list(self.network.graph.neighbors(node))

                # 处理坐标转换
                if self.topology == 'lattice':
                    # 对于2D格子网络，需要将元组坐标转换为防御者索引
                    if isinstance(node, tuple):
                        node_idx = node[0] * int(self.N ** 0.5) + node[1]
                        neighbor_indices = [n[0] * int(self.N ** 0.5) + n[1] for n in neighbors]
                    else:
                        node_idx = node
                        neighbor_indices = neighbors
                else:
                    node_idx = node
                    neighbor_indices = neighbors

                group = [self.defenders[node_idx]] + [self.defenders[n] for n in neighbor_indices]
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
                if self.topology == 'lattice':
                    # 对于2D格子网络，需要将防御者的整数ID转换为坐标
                    L = int(self.N ** 0.5)
                    coord = (d.id // L, d.id % L)
                    neighbor_coords = list(self.network.graph.neighbors(coord))
                    if neighbor_coords:
                        neighbor_coord = random.choice(neighbor_coords)
                        neighbor_id = neighbor_coord[0] * L + neighbor_coord[1]
                        fermi_update(d, self.defenders[neighbor_id], self.K)
                else:
                    # 对于其他网络拓扑，直接使用整数ID
                    neighbor_ids = list(self.network.graph.neighbors(d.id))
                    if neighbor_ids:
                        neighbor_id = random.choice(neighbor_ids)
                        fermi_update(d, self.defenders[neighbor_id], self.K)

            # === 5. 记录数据 ===
            coop_rate = sum(d.strategy == 'C' for d in self.defenders) / len(self.defenders)
            recorder.record(coop_rate, local_succ, self.attacker.q)

            # === 6. 重置收益 ===
            for d in self.defenders:
                d.reset_payoff()
