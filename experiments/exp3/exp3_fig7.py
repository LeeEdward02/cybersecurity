# experiments/exp3_fig7_re.py
"""
实验3.2：Fig.7 独立复现脚本
--------------------------
功能：仅复现论文Fig.7（4类拓扑的合作率与攻击率时间演化）
实验参数（匹配论文）：
- 节点数N=1600（40×40晶格，其他拓扑一致）
- 博弈参数：r=6.0（高于临界值）、q=0.4（固定攻击概率）
- 仿真轮次：2000轮（前1000轮暂态，后1000轮稳态）
- 初始状态：50%合作者，50%叛逃者
- 策略更新：Fermi规则（K=0.1）
"""
import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# 解决核心模块导入路径（确保能找到core文件夹）
current_file = os.path.abspath(__file__)
experiments_dir = os.path.dirname(current_file)
project_root = os.path.dirname(experiments_dir)
sys.path.append(project_root)

# 导入核心模块（仅保留Fig.7必需的类）
from core.agents import Defender, Attacker
from core.games import PublicGoodsGame, DefenderAttackerGame
from core.topology import NetworkTopology
from core.evolution import fermi_update
from core.recorder import DataRecorder


class TopologyImpactSimulation:
    """仿真初始化"""
    def __init__(self, topology_type, N=1600, r=6.0, q0=0.4, K=0.1):
        # 1. 基础参数初始化
        self.topology_type = topology_type
        self.N = N
        self.r = r
        self.q0 = q0
        self.K = K
        self.rounds = 2000  # 总仿真轮次
        self.transient = 1000  # 暂态轮次

        # 2. 生成网络拓扑
        self._init_topology()

        # 3. 初始化防御者（50%初始合作率）
        self._init_defenders()

        # 4. 初始化博弈实例与攻击者（无反馈，q固定0.4）
        self.attacker = Attacker(q0=q0, alpha=0.0)  # alpha=0→无反馈
        self.pgg = PublicGoodsGame(r=r, mu=40)      # PGG投资成本mu=40
        self.dag = DefenderAttackerGame(            # DAG收益矩阵（论文Table1）
            gamma1=50, gamma2=10, delta=50, d=50, c=10
        )

        # 5. 绑定防御者邻居（适配晶格坐标）
        self._bind_neighbors()

    def _init_topology(self):
        """生成4类拓扑（仅保留必需参数）"""
        topo_params = {
            'lattice': {},
            'smallworld': {'rewire_p': 0.08},  # 小世界重连概率0.08
            'scalefree': {'m': 3},             # 无标度网络m=3
            'random': {'p': 0.01}              # 随机网络边概率0.01
        }
        self.network = NetworkTopology(
            topology=self.topology_type,
            N=self.N,
            params=topo_params[self.topology_type]
        )

    def _init_defenders(self):
        """初始化防御者：50%合作（C），50%叛逃（D）"""
        self.defenders = []
        coop_count = self.N // 2  # 800个合作者
        for idx in range(self.N):
            strategy = 'C' if idx < coop_count else 'D'
            self.defenders.append(Defender(agent_id=idx, strategy=strategy))

    def _bind_neighbors(self):
        """为防御者绑定邻居（处理晶格坐标ID转换）"""
        if self.topology_type == 'lattice':
            L = int(self.N ** 0.5)  # 晶格边长40
            for d in self.defenders:
                coord = (d.id // L, d.id % L)  # ID→坐标
                neighbor_coords = self.network.get_neighbors(coord)
                d.neighbors = [self.defenders[nc[0]*L + nc[1]] for nc in neighbor_coords]
        else:
            for d in self.defenders:
                neighbor_ids = self.network.get_neighbors(d.id)
                d.neighbors = [self.defenders[nid] for nid in neighbor_ids]

    def run(self, recorder):
        """执行2000轮仿真，仅记录Fig.7所需数据（合作率、攻击率）"""
        print(f"📌 {self._get_fullname()} 开始仿真（r={self.r}）...")
        for t in range(self.rounds):
            # 1. 重置所有防御者收益
            for d in self.defenders:
                d.reset_payoff()

            # 2. 执行PGG博弈（5人小组：自身+4邻居）
            for d in self.defenders:
                group = [d] + d.neighbors[:4]  # 确保小组规模=5
                self.pgg.play(group)

            # 3. 执行DAG博弈（统计攻击成功率）
            attack_success = 0
            for d in self.defenders:
                focal_group = [d] + d.neighbors[:4]
                dp, _ = self.dag.play(d, self.attacker, focal_group)
                d.payoff += dp
                if dp < 0:  # 攻击成功判定（叛逃且被攻击）
                    attack_success += 1
            attack_rate = attack_success / self.N

            # 4. 防御者策略更新（Fermi规则）
            for d in self.defenders:
                if d.neighbors:
                    neighbor = random.choice(d.neighbors)
                    fermi_update(d, neighbor, self.K)

            # 5. 记录Fig.7必需数据（合作率、攻击率、q）
            coop_rate = sum(1 for d in self.defenders if d.strategy == 'C') / self.N
            avg_pay = sum(d.payoff for d in self.defenders) / self.N
            recorder.record(coop_rate, attack_rate, self.attacker.q, avg_pay)

        print(f"✅ {self._get_fullname()} 仿真完成\n")
        return recorder  # 返回记录器供绘图使用

    def _get_fullname(self):
        """返回拓扑全称（用于日志和图例）"""
        name_map = {
            'lattice': '2D-Lattice（晶格）',
            'smallworld': 'Small-World（小世界）',
            'scalefree': 'Scale-Free（无标度）',
            'random': 'Erdős-Rényi（随机）'
        }
        return name_map[self.topology_type]


def plot_fig7(all_recorders, save_path="exp3_fig7_evolution.png"):
    """绘制Fig.7：合作率与攻击率时间演化曲线（核心绘图函数）"""
    # 1. 定义拓扑样式（颜色+线型+标记，确保论文级区分度）
    styles = {
        'lattice': {'color': 'red', 'ls': '-', 'lw': 2, 'marker': 'o', 'markevery': 100, 'label': '2D-Lattice'},
        'smallworld': {'color': 'blue', 'ls': '--', 'lw': 2, 'marker': 's', 'markevery': 100, 'label': 'Small-World'},
        'scalefree': {'color': 'green', 'ls': '-.', 'lw': 2, 'marker': '^', 'markevery': 100, 'label': 'Scale-Free'},
        'random': {'color': 'orange', 'ls': ':', 'lw': 2, 'marker': 'd', 'markevery': 100, 'label': 'Erdős-Rényi'}
    }

    # 2. 创建1行2列子图（左：合作率，右：攻击率）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    fig.suptitle("Fig.7: Time Evolution of Cooperation and Attack Success (r=6.0, q=0.4)", 
                 fontsize=14, fontweight='bold', y=1.02)

    # ------------------------------
    # 子图1：合作率演化曲线
    # ------------------------------
    # 标记暂态结束线（1000轮）
    ax1.axvline(x=1000, color='gray', linestyle='--', alpha=0.5, label='Transient End (1000 rounds)')
    # 遍历拓扑绘制曲线
    for topo, rec in all_recorders.items():
        coop_rates = rec.records['coop_rate']  # 提取每轮合作率
        ax1.plot(range(2000), coop_rates, **styles[topo])
    # 子图1格式配置
    ax1.set_xlabel("Simulation Rounds", fontsize=12)
    ax1.set_ylabel("Cooperation Level", fontsize=12)
    ax1.set_ylim(0, 1.05)  # 合作率范围[0,1]，留少量余量
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(alpha=0.3)
    ax1.tick_params(axis='both', labelsize=10)

    # ------------------------------
    # 子图2：攻击率演化曲线
    # ------------------------------
    # 标记暂态结束线（与左图对齐）
    ax2.axvline(x=1000, color='gray', linestyle='--', alpha=0.5, label='Transient End (1000 rounds)')
    # 遍历拓扑绘制曲线
    for topo, rec in all_recorders.items():
        attack_rates = rec.records['attack_success_rate']  # 提取每轮攻击率
        ax2.plot(range(2000), attack_rates, **styles[topo])
    # 子图2格式配置
    ax2.set_xlabel("Simulation Rounds", fontsize=12)
    ax2.set_ylabel("Successful Attack Rate", fontsize=12)
    ax2.set_ylim(0, 0.5)  # 攻击率上限0.5（符合论文结果）
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(alpha=0.3)
    ax2.tick_params(axis='both', labelsize=10)

    # 调整布局并保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Fig.7 已保存至：{os.path.abspath(save_path)}")


def main():
    """Fig.7复现主流程：仅执行4类拓扑仿真+绘图"""
    # 1. 固定随机种子（确保结果可复现）
    random.seed(42)
    np.random.seed(42)

    # 2. 定义待仿真的4类拓扑
    topologies = ['lattice', 'smallworld', 'scalefree', 'random']
    all_recorders = {}  # 存储各拓扑的记录器

    # 3. 依次执行4类拓扑仿真
    for topo in topologies:
        # 初始化仿真对象
        sim = TopologyImpactSimulation(topology_type=topo)
        # 初始化数据记录器（仅记录2000轮数据）
        recorder = DataRecorder()
        # 执行仿真并保存记录器
        all_recorders[topo] = sim.run(recorder)

    # 4. 绘制并保存Fig.7
    plot_fig7(all_recorders)
    print("\n🎉 Fig.7 复现完成！")


if __name__ == "__main__":
    main()