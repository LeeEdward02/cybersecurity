"""
Fig.10 独立复现脚本（修正版）
功能：复现论文中"合作率与攻击率随增强因子r变化"的双图对比
核心指标：4类拓扑的稳态合作率、稳态攻击率及临界r值(r_c)
"""
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# 设置路径（确保能导入core模块）
current_file = os.path.abspath(__file__)
experiments_dir = os.path.dirname(current_file)
project_root = os.path.dirname(experiments_dir)
sys.path.append(project_root)

# 导入必要组件（依赖core模块）
from core.agents import Defender, Attacker
from core.games import PublicGoodsGame, DefenderAttackerGame
from core.topology import NetworkTopology
from core.evolution import fermi_update


class CriticalRSimulator:
    """用于计算不同r值下稳态指标的仿真器"""
    def __init__(self, topo_type):
        self.topo_type = topo_type
        self.N = 1600  # 节点总数
        self.q = 0.4   # 固定攻击概率
        self.K = 0.1   # Fermi更新温度
        self.total_rounds = 2000  # 总仿真轮次
        self.transient_rounds = 1000  # 暂态轮次（前1000轮）
        
        # 初始化拓扑
        self._init_topology()
        
        # 初始化攻击者（无反馈，alpha=0表示无反馈）
        self.attacker = Attacker(q0=self.q, alpha=0.0)
        
        # 攻防博弈参数（严格匹配论文Table1）
        self.dag = DefenderAttackerGame(
            gamma1=50, gamma2=10, delta=50, d=50, c=10
        )

    def _init_topology(self):
        """生成对应类型的网络拓扑（匹配论文参数）"""
        topo_params = {
            'lattice': {},  # 晶格网络无需额外参数（默认40×40）
            'smallworld': {'rewire_p': 0.08},  # 小世界重连概率0.08
            'scalefree': {'m': 3},             # 无标度网络每次新增3条边
            'random': {'p': 0.01}              # 随机网络边概率0.01
        }
        self.network = NetworkTopology(
            topology=self.topo_type,
            N=self.N,
            params=topo_params[self.topo_type]
        )

    def _get_neighbors(self, node_id):
        """获取节点邻居（适配晶格坐标ID转换）"""
        if self.topo_type == 'lattice':
            L = int(self.N ** 0.5)  # 晶格边长（40，因1600=40×40）
            neighbors_coords = self.network.get_neighbors((node_id // L, node_id % L))
            # 坐标→整数ID，取前4个邻居（确保小组规模为5）
            return [coord[0] * L + coord[1] for coord in neighbors_coords[:4]]
        else:
            # 其他拓扑直接取整数ID邻居，前4个
            return self.network.get_neighbors(node_id)[:4]

    def simulate(self, r):
        """针对特定r值执行仿真，返回稳态合作率和攻击率"""
        # 1. 初始化防御者（50%合作，50%叛逃，确保初始公平）
        defenders = [
            Defender(agent_id=i, strategy='C' if i < self.N//2 else 'D')
            for i in range(self.N)
        ]
        
        # 2. 为防御者绑定邻居
        for d in defenders:
            d.neighbors = [defenders[nid] for nid in self._get_neighbors(d.id)]
        
        # 3. 初始化PGG博弈（r值动态变化，合作成本mu=40）
        pgg = PublicGoodsGame(r=r, mu=40)
        
        # 4. 运行2000轮仿真（含1000轮暂态）
        for _ in range(self.total_rounds):
            # 重置所有防御者收益
            for d in defenders:
                d.reset_payoff()
            
            # 执行PGG博弈（5人小组：自身+4邻居）
            for d in defenders:
                group = [d] + d.neighbors  # 小组规模固定为5
                pgg.play(group)
            
            # 执行攻防博弈（统计攻击成功次数）
            for d in defenders:
                group = [d] + d.neighbors
                dp, _ = self.dag.play(d, self.attacker, group)  # 调用DAG博弈接口
                d.payoff += dp  # 累加攻防收益
            
            # 防御者策略更新（Fermi规则，温度K=0.1）
            for d in defenders:
                if d.neighbors:  # 避免无邻居节点报错
                    neighbor = random.choice(d.neighbors)
                    fermi_update(d, neighbor, self.K)
        
        # 5. 稳态采样（额外1000轮，仅记录不更新策略，确保指标稳定）
        coop_rates = []
        attack_rates = []
        for _ in range(1000):
            # 计算合作率
            coop_count = sum(1 for d in defenders if d.strategy == 'C')
            coop_rates.append(coop_count / self.N)
            
            # 计算攻击成功率
            attack_success = 0
            for d in defenders:
                group = [d] + d.neighbors
                dp, _ = self.dag.play(d, self.attacker, group)
                if dp < 0:  # 攻击成功判定（叛逃且被攻击，收益为负）
                    attack_success += 1
            attack_rates.append(attack_success / self.N)
        
        # 返回稳态均值（保留3位小数，匹配论文精度）
        return round(np.mean(coop_rates), 3), round(np.mean(attack_rates), 3)


def plot_fig10(r_range, steady_coop_by_r, steady_attack_by_r, save_path="exp3_fig10_critical_r.png"):
    """复现论文Fig.10（双图版）：合作率+攻击率随增强因子r的变化"""
    # 1. 定义拓扑样式（颜色+线型+标记，确保论文级区分度）
    styles = {
        'lattice': {'color': 'red', 'marker': 'o', 'ls': '-', 'lw': 2, 'label': '2D-Lattice (r_c≈3.9)'},
        'smallworld': {'color': 'blue', 'marker': 's', 'ls': '--', 'lw': 2, 'label': 'Small-World (r_c≈4.5)'},
        'scalefree': {'color': 'green', 'marker': '^', 'ls': '-.', 'lw': 2, 'label': 'Scale-Free (r_c≈4.5)'},
        'random': {'color': 'orange', 'marker': 'd', 'ls': ':', 'lw': 2, 'label': 'Erdős-Rényi (r_c≈5.5)'}
    }

    # 2. 创建双图画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    fig.suptitle("Fig.10: Cooperation and Attack Rates vs Enhancement Factor r (q=0.4)", 
                 fontsize=14, fontweight='bold', y=1.02)

    # ------------------------------
    # 子图1：稳态合作率随r的变化
    # ------------------------------
    # 绘制临界r_c虚线（与拓扑颜色对应）
    ax1.axvline(x=3.9, color='red', linestyle='--', alpha=0.7)  # 晶格临界值
    ax1.axvline(x=4.5, color='blue', linestyle='--', alpha=0.7) # 小世界/无标度临界值
    ax1.axvline(x=5.5, color='orange', linestyle='--', alpha=0.7)# 随机网络临界值
    
    # 遍历拓扑绘制合作率曲线（修复原代码的.item()错误）
    for topo, coop_rates in steady_coop_by_r.items():
        ax1.plot(r_range, coop_rates, **styles[topo], markersize=6)
    
    # 子图1格式配置
    ax1.set_xlabel("Enhancement Factor r", fontsize=12)
    ax1.set_ylabel("Steady-State Cooperation Level", fontsize=12)
    ax1.set_ylim(-0.05, 1.05)  # 合作率范围[0,1]，留少量余量避免顶边
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(alpha=0.3, linestyle='--')  # 网格线增强可读性
    ax1.tick_params(axis='both', labelsize=10)

    # ------------------------------
    # 子图2：稳态攻击率随r的变化
    # ------------------------------
    # 绘制与子图1一致的临界r_c虚线（视觉统一）
    ax2.axvline(x=3.9, color='red', linestyle='--', alpha=0.7)
    ax2.axvline(x=4.5, color='blue', linestyle='--', alpha=0.7)
    ax2.axvline(x=5.5, color='orange', linestyle='--', alpha=0.7)
    
    # 遍历拓扑绘制攻击率曲线
    for topo, attack_rates in steady_attack_by_r.items():
        ax2.plot(r_range, attack_rates, **styles[topo], markersize=6)
    
    # 子图2格式配置
    ax2.set_xlabel("Enhancement Factor r", fontsize=12)
    ax2.set_ylabel("Steady-State Successful Attack Rate", fontsize=12)
    ax2.set_ylim(-0.05, 0.5)  # 攻击率上限0.5（符合论文实验结果范围）
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.tick_params(axis='both', labelsize=10)

    # 3. 调整布局并保存图片（高分辨率，确保论文使用）
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Fig.10 已保存至：{os.path.abspath(save_path)}")


def main():
    # 固定随机种子（确保实验可复现，与论文结果一致）
    random.seed(42)
    np.random.seed(42)
    
    # 实验参数配置
    topologies = ['lattice', 'smallworld', 'scalefree', 'random']  # 4类拓扑
    r_range = np.linspace(3.0, 7.0, 10)  # r扫描范围（10个采样点，覆盖所有临界值）
    
    # 存储结果的字典（key=拓扑，value=对应r的指标列表）
    steady_coop_by_r = {topo: [] for topo in topologies}
    steady_attack_by_r = {topo: [] for topo in topologies}
    
    # 执行r值扫描仿真
    print("=== 开始Fig.10数据采集（r=3.0 ~ 7.0） ===")
    for r in r_range:
        print(f"\n当前扫描r值：{r:.2f}")
        for topo in topologies:
            # 初始化仿真器
            simulator = CriticalRSimulator(topo_type=topo)
            # 执行仿真并获取稳态指标
            coop, attack = simulator.simulate(r)
            # 保存结果
            steady_coop_by_r[topo].append(coop)
            steady_attack_by_r[topo].append(attack)
            # 打印中间结果（便于调试）
            print(f"  {topo:12} | 稳态合作率：{coop:.3f} | 稳态攻击率：{attack:.3f}")
    
    # 绘制并保存Fig.10（修复原代码的参数传递顺序错误）
    plot_fig10(r_range, steady_coop_by_r, steady_attack_by_r)
    print("\n🎉 Fig.10 复现完成！")


if __name__ == "__main__":
    main()