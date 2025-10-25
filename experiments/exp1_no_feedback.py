# experiments/exp1_no_feedback.py
"""
实验1：无反馈机制实验
--------------------
实验目标：
1. 研究缺乏攻击者反馈时，固定攻击概率对合作的影响
2. 确定合作水平发生不连续转变的临界增强因子 rc
3. 证明外部风险（攻击概率 q）促进防御者合作

实验设计：
- 网络拓扑：2D-Lattice (40x40 = 1600节点)
- 交互规模：k=4邻居，小组规模N(v)=k+1=5
- 固定参数：alphaA=0.0 (无反馈), q=0.4 (固定攻击概率)
- 变化参数：r (公共物品增强因子，系统性变化)
- 初始状态：50%合作者，50%叛逃者
- 运行参数：2000轮，确保达到稳定状态
"""

import numpy as np
from core.simulation import CyberSecuritySimulation
from core.recorder import DataRecorder
from core.evolution import fermi_update


class NoFeedbackSimulation(CyberSecuritySimulation):
    """
    无反馈机制实验仿真类
    --------------------
    实现空间公共物品博弈(SPGG) + 防御者-攻击者博弈(DAG)的复合模型

    实验条件：
    - 网络拓扑：2D-Lattice (1600节点)
    - alphaA=0.0：攻击者不适应，q保持固定
    - q=0.4：固定攻击概率作为外部风险因子
    - k=4：每个节点4个邻居
    - 小组规模：N(v)=5 (中心节点+4邻居)
    - r：系统性变化的公共物品增强因子
    """

    def __init__(self, r=4.0, q=0.4, N=1600, rounds=4000, K=0.1):
        """
        初始化无反馈实验仿真

        Args:
            r (float): 公共物品增强因子 (系统性变化参数)
            q (float): 固定攻击概率 (默认0.4)
            N (int): 节点数量 (默认1600，40x40格子)
            rounds (int): 仿真轮数 (默认2000，确保达到稳定状态)
            K (float): Fermi更新温度 (默认0.1)
        """
        super().__init__(
            N=N,
            rounds=rounds,
            r=r,
            q0=q,
            alphaA=0.0,  # 核心设计：无反馈机制
            K=K,
            topology='lattice',
            params=None
        )

        # 验证网络参数
        assert int(N ** 0.5) ** 2 == N, "N必须是完全平方数以构成正方形格子"
        self.L = int(N ** 0.5)  # 格子边长

        # 验证拓扑符合要求
        assert self.topology == 'lattice', "实验要求使用2D-Lattice网络拓扑"

        # 验证初始策略分布
        initial_coop = sum(d.strategy == 'C' for d in self.defenders) / len(self.defenders)
        assert 0.45 <= initial_coop <= 0.55, f"初始合作率应为50%，当前为{initial_coop:.2f}"

        # 固定攻击概率验证
        assert q == 0.4, f"实验要求固定攻击概率q=0.4，当前为{q}"

        print(f"无反馈实验初始化完成:")
        print(f"  网络规模: {self.L}x{self.L} = {N} 节点")
        print(f"  邻居数: k=4")
        print(f"  小组规模: N(v)=5")
        print(f"  固定攻击概率: q={q}")
        print(f"  无反馈机制: alphaA=0.0")
        print(f"  公共物品增强因子: r={r}")
        print(f"  初始合作率: {initial_coop:.2f}")

    def run(self, recorder):
        """
        执行无反馈机制下的仿真流程

        Args:
            recorder (DataRecorder): 数据记录对象
        """
        print(f"开始无反馈实验仿真 (r={self.r}, q={self.q0})...")

        for t in range(self.rounds):
            # === 1. 空间公共物品博弈 (SPGG) ===
            # 每个防御者与其4个邻居形成N(v)=5的小组
            for node in self.network.graph.nodes:
                neighbors = list(self.network.graph.neighbors(node))

                # 对于2D格子网络，需要将元组坐标转换为防御者索引
                if isinstance(node, tuple):
                    # 将(x,y)坐标转换为线性索引：index = x * L + y
                    node_idx = node[0] * self.L + node[1]
                    neighbor_indices = [n[0] * self.L + n[1] for n in neighbors]
                else:
                    node_idx = node
                    neighbor_indices = neighbors

                group = [self.defenders[node_idx]] + [self.defenders[n] for n in neighbor_indices]

                # 验证小组规模
                assert len(group) == 5, f"小组规模应为5，当前为{len(group)}"

                # 执行公共物品博弈
                self.pgg.play(group)

            # === 2. 防御者-攻击者博弈 (DAG) ===
            # 每个防御者面临固定攻击概率q的攻击
            attack_success, total_attacks = 0, 0
            for d in self.defenders:
                dp, ap = self.dag.play(d, self.attacker)
                d.payoff += dp

                # 记录攻击成功率
                if dp < 0:  # 攻击成功判定
                    attack_success += 1
                total_attacks += 1

            # === 3. 攻击者状态保持不变 (无反馈机制) ===
            # alphaA=0.0，攻击者不更新策略
            # q(t+1) = q(t) = q0 (保持固定)
            # 这里的update_feedback调用由于alphaA=0.0不会产生实际效果
            local_succ = attack_success / total_attacks if total_attacks > 0 else 0
            self.attacker.update_feedback(local_succ, local_succ)

            # 验证攻击概率保持固定
            assert abs(self.attacker.q - self.q0) < 1e-10, \
                f"攻击概率应保持{self.q0}，当前为{self.attacker.q}"

            # === 4. 策略更新 (Fermi规则) ===
            for d in self.defenders:
                # 随机选择一个邻居进行策略比较
                # 对于2D格子网络，需要将防御者ID转换为坐标
                if isinstance(d.id, int):
                    # 将线性索引转换为坐标：(x, y) = (id // L, id % L)
                    coord = (d.id // self.L, d.id % self.L)
                    neighbors = list(self.network.graph.neighbors(coord))
                    if neighbors:
                        neighbor_coord = random.choice(neighbors)
                        neighbor_id = neighbor_coord[0] * self.L + neighbor_coord[1]
                    else:
                        continue  # 如果没有邻居，跳过
                else:
                    # 如果ID已经是坐标形式
                    neighbors = list(self.network.graph.neighbors(d.id))
                    if neighbors:
                        neighbor_coord = random.choice(neighbors)
                        neighbor_id = neighbor_coord[0] * self.L + neighbor_coord[1]
                    else:
                        continue

                neighbor = self.defenders[neighbor_id]

                # 使用Fermi更新规则
                fermi_update(d, neighbor, self.K)

            # === 5. 数据记录 ===
            coop_rate = sum(d.strategy == 'C' for d in self.defenders) / len(self.defenders)
            attack_success_rate = attack_success / total_attacks if total_attacks > 0 else 0

            recorder.record(coop_rate, attack_success_rate, self.attacker.q, 0)

            # === 6. 重置本轮收益 ===
            for d in self.defenders:
                d.reset_payoff()

            # 进度显示
            if (t + 1) % 500 == 0:
                print(f"  第 {t+1}/{self.rounds} 轮: 合作率={coop_rate:.3f}, 攻击成功率={attack_success_rate:.3f}")

        # 最终统计
        final_coop = sum(d.strategy == 'C' for d in self.defenders) / len(self.defenders)
        print(f"仿真完成! 最终合作率: {final_coop:.3f}")


def run_exp1():
    """
    运行无反馈机制实验
    可以调整r值来观察合作相变
    """
    print("=== 实验1：无反馈机制 ===")
    print("研究固定攻击概率对合作演化的影响")

    # 可以测试不同的r值来观察临界相变
    test_r_values = [3.1]  # 系统性变化的增强因子

    for r in test_r_values:
        print(f"\n--- 测试 r={r} ---")
        sim = NoFeedbackSimulation(r=r, q=0.4)
        rec = DataRecorder()
        sim.run(rec)

        # 打印最终结果
        final_coop = rec.records['coop_rate'][-1]
        final_attack_rate = rec.records['attack_success_rate'][-1]
        final_attacker_q = rec.records['q_history'][-1]

        print(f"最终结果:")
        print(f"  合作率: {final_coop:.3f}")
        print(f"  攻击成功率: {final_attack_rate:.3f}")
        print(f"  攻击者概率: {final_attacker_q:.3f}")


if __name__ == "__main__":
    import random

    # 运行基本实验
    run_exp1()
