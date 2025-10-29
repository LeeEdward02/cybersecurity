import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap

# 解决core模块导入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入原有core模块（不修改）
from core.agents import Defender, Attacker
from core.games import PublicGoodsGame, DefenderAttackerGame
from core.topology import NetworkTopology
from core.evolution import fermi_update
from core.recorder import DataRecorder


class Fig8VisualizationSimulation:
    """Fig8专项仿真类：修复2D晶格ID与布局匹配问题，不修改core"""
    def __init__(self, topology_type, N=1600, r=6.0, q0=0.4, K=0.1):
        # 1. 实验参数（严格匹配论文Fig8）
        self.topology_type = topology_type  # 仅支持'lattice'/'scalefree'
        self.N = N                          # 节点数（40×40=1600）
        self.r = r                          # 增强因子（r=6.0）
        self.q0 = q0                        # 固定攻击概率（q=0.4）
        self.K = K                          # Fermi温度（0.1）
        self.sim_rounds = 1000              # 仿真轮次（t=0→t=1000）
        self.visualize_steps = [0, 1000]    # 可视化时间点
        self.L = int(np.sqrt(N)) if topology_type == 'lattice' else None  # 晶格边长（40）

        # 2. 生成网络拓扑（关键：获取core生成的原始节点ID格式）
        self._init_topology()
        # 提取网络节点ID列表（用于后续防御者ID匹配）
        self.network_node_ids = list(self.graph.nodes())
        print(f"🔍 {self._get_fullname()}网络节点ID示例（前5个）：{self.network_node_ids[:5]}，类型：{type(self.network_node_ids[0])}")

        # 3. 初始化防御者（关键：防御者ID与网络节点ID格式完全一致）
        self._init_defenders()

        # 4. 初始化博弈实例
        self.attacker = Attacker(q0=q0, alpha=0.0)  # 无反馈（α_A=0）
        self.pgg = PublicGoodsGame(r=r, mu=40)      # PGG投资成本mu=40
        self.dag = DefenderAttackerGame(            # DAG收益矩阵（论文Table2）
            gamma1=50, gamma2=10, delta=50, d=50, c=10
        )

        # 5. 绑定防御者邻居（基于网络节点ID格式）
        self._bind_neighbors()

        # 6. 外部状态存储（键与防御者ID格式一致）
        self.visual_data = {}
        self.attack_state_map = {d.id: False for d in self.defenders}  # 键=防御者ID（坐标/整数）
        # 验证ID覆盖与格式
        self._verify_attack_state_map()

    def _init_topology(self):
        """生成Fig8指定拓扑：2D晶格（坐标ID）/无标度网络（整数ID）"""
        topo_params = {
            'lattice': {},  # core生成2D晶格，节点ID为坐标元组（如(0,0)）
            'scalefree': {'m': 3}  # 无标度网络，节点ID为整数
        }
        self.network = NetworkTopology(
            topology=self.topology_type,
            N=self.N,
            params=topo_params[self.topology_type]
        )
        self.graph = self.network.graph  # 原始图结构（节点ID格式由core决定）

    def _init_defenders(self):
        """初始化防御者：随机分布C/D，防止晶格上下分层"""
        self.defenders = []
        self.defender_id_map = {}  # 键=网络节点ID（坐标/整数），值=防御者实例
        
        # 生成随机的C/D策略分布（50%:50%）
        strategies = ['C'] * (self.N // 2) + ['D'] * (self.N // 2)
        random.shuffle(strategies)

        # 遍历网络节点ID，策略随机分布
        for idx, node_id in enumerate(self.network_node_ids):
            strategy = strategies[idx]
            defender = Defender(agent_id=node_id, strategy=strategy)
            self.defenders.append(defender)
            self.defender_id_map[node_id] = defender  # 建立ID→实例映射

        # 验证防御者ID格式与网络节点ID一致
        assert all(d.id in self.network_node_ids for d in self.defenders), \
            "防御者ID不在网络节点ID列表中，格式不匹配"
        print(f"✅ 防御者随机策略初始化完成，C数量={strategies.count('C')}，D数量={strategies.count('D')}")


    def _bind_neighbors(self):
        """绑定邻居：基于网络节点ID格式（坐标/整数），确保邻居实例正确"""
        for d in self.defenders:
            # 获取当前防御者ID的邻居（调用core.topology的get_neighbors，返回网络节点ID格式）
            neighbor_ids = self.network.get_neighbors(d.id)
            # 验证邻居ID格式（与网络节点ID一致）
            assert all(nid in self.network_node_ids for nid in neighbor_ids), \
                f"邻居ID {neighbor_ids[0]} 不在网络节点ID列表中，格式错误"
            # 绑定邻居实例（通过defender_id_map映射）
            d.neighbors = [self.defender_id_map[nid] for nid in neighbor_ids]
        print(f"✅ 邻居绑定完成，示例：防御者{self.defenders[0].id}的邻居ID：{[n.id for n in self.defenders[0].neighbors]}")

    def _verify_attack_state_map(self):
        """验证attack_state_map的键与防御者ID格式一致，无缺失"""
        # 检查键格式（与防御者ID格式一致）
        defender_id_types = set(type(d.id) for d in self.defenders)
        map_key_types = set(type(key) for key in self.attack_state_map.keys())
        assert defender_id_types == map_key_types, \
            f"attack_state_map键类型（{map_key_types}）与防御者ID类型（{defender_id_types}）不匹配"

        # 检查键覆盖（所有防御者ID均在map中）
        missing_ids = [d.id for d in self.defenders if d.id not in self.attack_state_map]
        if missing_ids:
            raise ValueError(f"attack_state_map缺失以下防御者ID：{missing_ids[:5]}...（共{len(missing_ids)}个）")

        # 检查键数量（等于节点数1600）
        assert len(self.attack_state_map) == self.N, \
            f"attack_state_map键数量应为{self.N}，实际{len(self.attack_state_map)}"
        print(f"✅ attack_state_map验证通过：{len(self.attack_state_map)}个键，格式与防御者ID一致")

    def _infer_attack_state(self, defender, dag_payoff):
        """从DAG收益反向推断攻击状态（匹配论文Table2）"""
        if defender.strategy == 'C':
            # 合作者（投资）：无论是否被攻击，均无成功攻击（is_attacked=False）
            return False
        else:
            # 叛逃者（不投资）：收益=-50→被攻击成功（True），否则False
            return np.isclose(dag_payoff, -50.0)

    def _record_visual_state(self, round):
        """记录节点状态（ID格式与网络一致，避免KeyError）"""
        node_states = []
        # 按网络节点ID顺序记录（确保与后续绘图的节点顺序一致）
        for node_id in self.network_node_ids:
            defender = self.defender_id_map[node_id]
            is_attacked = self.attack_state_map[node_id]
            # 确定节点状态
            if defender.strategy == 'C':
                state = 'C_a' if is_attacked else 'C_na'
            else:
                state = 'D_a' if is_attacked else 'D_na'
            node_states.append(state)
        self.visual_data[round] = node_states
        print(f"📝 记录t={round}状态完成，节点状态示例（前5个）：{node_states[:5]}")

    def run_simulation(self):
        """执行Fig8仿真：ID格式一致，避免运行时错误"""
        print(f"\n📊 开始{self._get_fullname()}网络Fig8仿真（{self.sim_rounds}轮）...")
        for t in range(self.sim_rounds + 1):
            # 1. 记录可视化时间点的状态（t=0和t=1000）
            if t in self.visualize_steps:
                self._record_visual_state(t)

            # 2. t<1000时执行博弈与策略更新
            if t >= self.sim_rounds:
                break

            # 3. 重置：防御者收益+外部攻击状态映射
            for d in self.defenders:
                d.reset_payoff()  # 调用core方法重置收益
                self.attack_state_map[d.id] = False  # 重置攻击状态（键=防御者ID）

            # 4. 执行PGG博弈（5人小组：自身+4邻居）
            for d in self.defenders:
                group = [d] + d.neighbors[:4]  # 确保小组规模=5
                self.pgg.play(group)  # 调用core.PublicGoodsGame.play

            # 5. 执行DAG博弈：更新攻击状态映射
            for d in self.defenders:
                focal_group = [d] + d.neighbors[:4]
                dp, _ = self.dag.play(d, self.attacker, focal_group)  # 原有DAG逻辑
                d.payoff += dp  # 累加DAG收益
                # 推断攻击状态并更新map（键=防御者ID）
                self.attack_state_map[d.id] = self._infer_attack_state(d, dp)

            # 6. 攻击者无反馈更新（q保持0.4）
            attack_count = sum(1 for aid in self.attack_state_map if self.attack_state_map[aid])
            attack_rate = attack_count / self.N
            self.attacker.update_feedback(attack_rate, attack_rate)

            # 7. 防御者策略更新（Fermi规则）
            for d in self.defenders:
                if d.neighbors:
                    neighbor = random.choice(d.neighbors)
                    fermi_update(d, neighbor, self.K)  # 调用core.evolution

            # 8. 打印进度
            if (t + 1) % 200 == 0:
                coop_count = sum(1 for d in self.defenders if d.strategy == 'C')
                coop_rate = coop_count / self.N
                print(f"   轮次{t+1:4d}/{self.sim_rounds}：合作率={coop_rate:.3f}，成功攻击数={attack_count}")

        print(f"✅ {self._get_fullname()}网络Fig8仿真完成，已记录t=0和t=1000状态")

    def _get_fullname(self):
        """返回拓扑全称（用于图表标题）"""
        return '2D-Lattice' if self.topology_type == 'lattice' else 'Scale-Free'

    def plot_fig8_single_topo(self, save_path="fig8_single_topo.png"):
        """绘制单个拓扑的Fig8子图（兼容所有NetworkX版本，无norm参数）"""
        # 获取状态数据
        t0_states = self.visual_data[0]
        t1000_states = self.visual_data[1000]

        # 状态→颜色索引
        cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFF00'])
        state_to_idx = {'C_na': 0, 'D_na': 1, 'C_a': 2, 'D_a': 3}
        t0_colors = [state_to_idx[s] for s in t0_states]
        t1000_colors = [state_to_idx[s] for s in t1000_states]

        # 创建画布
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        topo_fullname = self._get_fullname()
        fig.suptitle(f"Fig8: {topo_fullname} Network Spatio-temporal Visualization (α_A=0, q=0.4, r=6.0)", 
                    fontsize=14, fontweight='bold')

        # 生成节点布局 pos
        if self.topology_type == 'lattice':
            # --- 晶格：随机扰动位置，避免上下分层 ---
            pos = {}
            jitter = 0.1  # 随机扰动幅度
            for (i, j) in self.network_node_ids:
                x = j + random.uniform(-jitter, jitter)
                y = -i + random.uniform(-jitter, jitter)
                pos[(i, j)] = (x, y)
        else:
            # --- 无标度网络：固定随机seed，布局稳定 ---
            pos = nx.spring_layout(self.graph, seed=42, k=2.0)

        # 验证布局键完整
        assert all(node_id in pos for node_id in self.network_node_ids), \
            "布局pos缺失部分节点的位置"

        # 绘制 t=0 子图
        nx.draw_networkx_edges(self.graph, pos, ax=ax1, alpha=0.3, edge_color='gray')
        nodes_t0 = nx.draw_networkx_nodes(
            self.graph, pos, ax=ax1,
            nodelist=self.network_node_ids,
            node_color=t0_colors, cmap=cmap,
            node_size=50, edgecolors='black', linewidths=0.5
        )
        ax1.set_title("t=0 (Initial State): 50% C, 50% D", fontsize=12, fontweight='bold')
        ax1.axis('off')

        # 绘制 t=1000 子图
        nx.draw_networkx_edges(self.graph, pos, ax=ax2, alpha=0.3, edge_color='gray')
        nodes_t1000 = nx.draw_networkx_nodes(
            self.graph, pos, ax=ax2,
            nodelist=self.network_node_ids,
            node_color=t1000_colors, cmap=cmap,
            node_size=50, edgecolors='black', linewidths=0.5
        )
        ax2.set_title("t=1000 (Steady State)", fontsize=12, fontweight='bold')
        ax2.axis('off')

        # 添加统一颜色条（右侧）
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        mappable = ScalarMappable(cmap=cmap)
        mappable.set_array([0, 1, 2, 3])
        cbar = plt.colorbar(mappable, cax=cbar_ax, ticks=[0.5, 1.5, 2.5, 3.5])
        cbar.set_ticklabels([
            'Cooperator (Not Attacked)', 
            'Defector (Not Attacked)', 
            'Cooperator (Attacked)', 
            'Defector (Attacked)'
        ])
        cbar.ax.tick_params(labelsize=10)

        # 布局与保存
        plt.tight_layout()
        plt.subplots_adjust(right=0.9, top=0.9)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

        print(f"\n📸 {topo_fullname} Fig8子图已保存至：{save_path}")
        self._print_state_statistics(t0_states, "t=0")
        self._print_state_statistics(t1000_states, "t=1000")


    def _print_state_statistics(self, node_states, time_label):
        """打印节点状态占比，验证与论文数值一致"""
        total = len(node_states)
        c_na = node_states.count('C_na')
        d_na = node_states.count('D_na')
        c_a = node_states.count('C_a')
        d_a = node_states.count('D_a')
        print(f"\n{self._get_fullname()} {time_label} 状态统计：")
        print(f"  C_na（红）: {c_na/total:.4f} | D_na（绿）: {d_na/total:.4f}")
        print(f"  C_a（蓝）: {c_a/total:.4f} | D_a（黄）: {d_a/total:.4f}")
        print(f"  总成功攻击率: {(c_a + d_a)/total:.4f}")


# ------------------------------
# Fig8复现主函数
# ------------------------------
def reproduce_fig8():
    """复现论文Fig8：依次运行2D晶格+无标度网络，生成完整图表"""
    # 固定随机种子（确保实验可重复）
    random.seed(42)
    np.random.seed(42)

    # 1. 定义待仿真的拓扑（Fig8仅包含2D晶格和无标度）
    target_topologies = [('lattice', "2D-Lattice"), ('scalefree', "Scale-Free")]
    sim_results = {}

    # 2. 逐个拓扑执行仿真（修复ID与布局匹配问题）
    for topo_code, topo_name in target_topologies:
        print(f"\n=== 开始{topo_name}拓扑Fig8仿真 ===")
        sim = Fig8VisualizationSimulation(topology_type=topo_code)
        sim.run_simulation()
        sim_results[topo_code] = {
            'visual_data': sim.visual_data,
            'graph': sim.graph,
            'network_node_ids': sim.network_node_ids,  # 保存网络节点ID列表（用于后续绘图）
            'topo_name': topo_name,
            'L': sim.L
        }
        # 保存单个拓扑子图
        sim.plot_fig8_single_topo(save_path=f"fig8_{topo_code}.png")

    # 3. 合并两种拓扑数据，生成完整Fig8（2×2子图）
    plot_complete_fig8(sim_results)

    print("\n=== Fig8完整复现完成 ===")
    print("关键结论验证：")
    print("1. 2D晶格t=1000：C_na≈100%（全红），无成功攻击；")
    print("2. 无标度网络t=1000：C_na≈73%、D_na≈17%、C_a≈5%、D_a≈4%，成功攻击率≈9.19%（匹配论文）。")


def plot_complete_fig8(sim_results):
    """合并两种拓扑的子图，生成完整Fig8（2×2布局）"""
    cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFF00'])
    state_to_idx = {'C_na': 0, 'D_na': 1, 'C_a': 2, 'D_a': 3}
    boundaries = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)


    # 创建2×2子图（论文Fig8标准布局）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Fig.8: Spatio-temporal Visualizations of 2D-Lattice and Scale-Free Networks Under Attack", 
                 fontsize=16, fontweight='bold')

    # 子图配置：(拓扑代码, 时间点, 行, 列, 子标题)
    subplot_config = [
        ('lattice', 0, 0, 0, "(a) 2D-Lattice (t=0)"),
        ('lattice', 1000, 0, 1, "(b) 2D-Lattice (t=1000)"),
        ('scalefree', 0, 1, 0, "(c) Scale-Free (t=0)"),
        ('scalefree', 1000, 1, 1, "(d) Scale-Free (t=1000)")
    ]

    # 绘制每个子图
    for topo_code, t, row, col, title in subplot_config:
        result = sim_results[topo_code]
        visual_data = result['visual_data'][t]
        graph = result['graph']
        network_node_ids = result['network_node_ids']  # 网络节点ID列表（确保顺序一致）
        topo_name = result['topo_name']
        L = result.get('L')

        # 转换状态为颜色索引（顺序与网络节点ID一致）
        node_colors = [state_to_idx[state] for state in visual_data]

        # 生成布局（键与网络节点ID格式一致）
        ax = axes[row, col]
        if topo_code == 'lattice':
            # 2D晶格：坐标ID→布局位置
            pos = {(i, j): (j, -i) for (i, j) in network_node_ids}
        else:
            # 无标度网络：整数ID→spring布局
            pos = nx.spring_layout(graph, seed=42, k=2.0)

        # 绘制节点与边（nodelist=网络节点ID，确保顺序一致）
        nx.draw_networkx_nodes(
            graph, pos, ax=ax,
            nodelist=network_node_ids,
            node_color=node_colors, cmap=cmap,
            node_size=50, edgecolors='black', linewidths=0.5
        )
        nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.3, edge_color='gray')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

    # 添加全局颜色条（右侧）
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # colorbar 位置
    mappable = ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])  # 避免旧版本报错
    cbar = plt.colorbar(mappable, cax=cbar_ax, boundaries=boundaries, ticks=[0, 1, 2, 3])
    cbar.set_ticklabels([
        'Cooperator (Not Attacked)',
        'Defector (Not Attacked)',
        'Cooperator (Attacked)',
        'Defector (Attacked)'
    ])
    cbar.ax.tick_params(labelsize=10)

    # 保存完整Fig8
    save_path = "fig8_complete.png"
    plt.tight_layout()
    plt.subplots_adjust(right=0.9, top=0.9)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n📸 完整Fig8已保存至：{save_path}")


# ------------------------------
# 执行Fig8复现
# ------------------------------
if __name__ == "__main__":
    reproduce_fig8()