import random
import matplotlib.pyplot as plt
import numpy as np  # 用于数值计算
from scipy.interpolate import make_interp_spline  # 用于曲线插值
from core.simulation import CyberSecuritySimulation
from core.evolution import fermi_update
from core.games import DefenderAttackerGame

class ExtendedFeedbackSimulation(CyberSecuritySimulation):
    def run(self, recorder, current_alphaA):
        print(f"start simulation (alphaA={self.alphaA}, q0={self.q0}, r={self.r})...")
        warmup_rounds = 200  # 预热轮次：排除初始波动
        total_rounds = self.rounds + warmup_rounds
        
        for t in range(total_rounds):
            # 1. 空间公共物品博弈（不变）
            for defender in self.defenders:
                group = [defender] + defender.neighbors
                self.pgg.play(group)
            
            # 2. 攻防博弈（不变，确保收益计算正确）
            attack_success = 0
            total_attacks = 0
            defender_total_payoff = 0
            attacker_total_payoff = 0
            
            for d in self.defenders:
                focal_group = [d] + d.neighbors
                dp, ap = self.dag.play(d, self.attacker, focal_group)
                d.payoff += dp
                defender_total_payoff += dp
                attacker_total_payoff += ap
                
                if dp < 0:
                    attack_success += 1
                total_attacks += 1
            
            # 3. 攻击者反馈更新（不变，无需记录q）
            local_success = attack_success / total_attacks if total_attacks > 0 else 0
            global_success = local_success
            self.attacker.q += self.alphaA * (local_success - global_success)
            self.attacker.q = max(0.01, min(0.99, self.attacker.q))
            
            # 4. 防御者策略更新（不变）
            for d in self.defenders:
                if d.neighbors:
                    neighbor = random.choice(d.neighbors)
                    dynamic_K = self.K * (5 / self.r) if self.r > 0 else self.K
                    fermi_update(d, neighbor, dynamic_K)
            
            # 5. 数据记录（关键：仅传子图a-d所需数据，移除q）
            coop_rate = sum(d.strategy == 'C' for d in self.defenders) / len(self.defenders)
            attack_success_rate = attack_success / total_attacks if total_attacks > 0 else 0
            
            if t >= warmup_rounds:
                avg_def_payoff = defender_total_payoff / len(self.defenders)
                avg_att_payoff = attacker_total_payoff / len(self.defenders)
                # 调用修改后的record_param：无q参数
                recorder.record_param(
                    alphaA=current_alphaA,
                    coop=coop_rate,
                    attack=attack_success_rate,
                    def_payoff=avg_def_payoff,
                    att_payoff=avg_att_payoff
                )
            
            # 6. 重置收益（不变）
            for d in self.defenders:
                d.reset_payoff()
            
            # 进度显示（不变）
            if (t + 1) % 500 == 0:
                print(f"the {t+1}/{total_rounds} times : cooperation={coop_rate:.3f}, "
                    f"Successful Attack={attack_success_rate:.3f}, Attack probablity={self.attacker.q:.3f}")




def run_figure6_experiment():
    """运行图6的实验：系统随alpha的变化（固定r=4.5，仅保留子图a-d）"""
    print("\n=== 图6实验：系统随alpha的变化 ===")
    # 1. 实验参数配置（平衡速度与数据有效性）
    alpha_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]  # 关键临界值0.2保留
    q0_values = [0.25, 0.30, 0.40, 0.50, 0.60]  # 与PDF一致的初始攻击概率
    r = 4.5  # 固定增强因子（PDF核心参数）
    rounds = 1000  # 确保稳态数据充足，避免收益计算异常
    repeat_times = 4  # 减少随机波动，保证收益数据平滑
    recorders = {q0: DataRecorder() for q0 in q0_values}

    # 2. 实验循环（生成数据：修复收益计算的数据源）
    for alphaA in alpha_values:
        print(f"\n--- 测试 alphaA={alphaA} ---")
        for q0 in q0_values:
            for _ in range(repeat_times):
                sim = ExtendedFeedbackSimulation(
                    alphaA=alphaA,
                    q0=q0,
                    r=r,
                    rounds=rounds
                )
                # 关键：传入current_alphaA，确保收益数据与alphaA正确关联
                sim.run(recorders[q0], current_alphaA=alphaA)

    # 3. 处理平均数据（确保收益数据与params维度严格一致）
    avg_data = {}
    for q0, recorder in recorders.items():
        data = recorder.get_averaged_data()
        # 强制对齐维度：仅保留有完整收益数据的alphaA（排除空值）
        valid_indices = [i for i, payoff in enumerate(data['defender_payoff']) if payoff != 0.0 or data['attacker_payoff'][i] != 0.0]
        if valid_indices:
            avg_data[q0] = {
                'params': [data['params'][i] for i in valid_indices],
                'coop_rate': [data['coop_rate'][i] for i in valid_indices],
                'attack_success_rate': [data['attack_success_rate'][i] for i in valid_indices],
                'defender_payoff': [data['defender_payoff'][i] for i in valid_indices],
                'attacker_payoff': [data['attacker_payoff'][i] for i in valid_indices]
            }
        else:
            # 无有效数据时，用原始params确保绘图不报错
            avg_data[q0] = data

    # 4. 绘图样式配置（确保曲线区分度，匹配目标图）
    styles = {
        0.25: {'color': '#e41a1c', 'marker': 'o', 'linewidth': 2.5, 'markersize': 7, 'label': 'q₀=0.25'},
        0.30: {'color': '#377eb8', 'marker': 's', 'linewidth': 2.5, 'markersize': 7, 'label': 'q₀=0.30'},
        0.40: {'color': '#4daf4a', 'marker': '^', 'linewidth': 2.5, 'markersize': 7, 'label': 'q₀=0.40'},
        0.50: {'color': '#984ea3', 'marker': 'D', 'linewidth': 2.5, 'markersize': 7, 'label': 'q₀=0.50'},
        0.60: {'color': '#ff7f00', 'marker': 'v', 'linewidth': 2.5, 'markersize': 7, 'label': 'q₀=0.60'}
    }

    # 5. 创建画布（简化为2行2列，仅保留子图a-d，避免子图e占用资源）
    plt.figure(figsize=(15, 12))  # 调整画布比例，适配2行2列
    critical_alpha = 0.20  # PDF理论临界值：alpha>0.2时系统突变

    # -------------------------- 子图(a)：防御者合作水平 --------------------------
    plt.subplot(2, 2, 1)
    for q0, data in avg_data.items():
        if len(data['params']) > 0:  # 仅绘制有数据的曲线
            plt.plot(
                data['params'], data['coop_rate'],
                color=styles[q0]['color'], marker=styles[q0]['marker'],
                linewidth=styles[q0]['linewidth'], markersize=styles[q0]['markersize'],
                label=styles[q0]['label']
            )
    plt.axvline(x=critical_alpha, color='black', linestyle='--', linewidth=2, label='临界α=0.2')
    plt.title('(a) 防御者合作水平', fontsize=13, fontweight='bold')
    plt.xlabel('攻击反馈速率α', fontsize=12)
    plt.ylabel('合作水平', fontsize=12)
    plt.xlim(0.04, 0.41)  # 匹配alpha_values范围，避免空白
    plt.ylim(0.0, 1.0)    # 合作水平理论范围（0-1）
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # -------------------------- 子图(b)：攻击成功率 --------------------------
    plt.subplot(2, 2, 2)
    for q0, data in avg_data.items():
        if len(data['params']) > 0:
            plt.plot(
                data['params'], data['attack_success_rate'],
                color=styles[q0]['color'], marker=styles[q0]['marker'],
                linewidth=styles[q0]['linewidth'], markersize=styles[q0]['markersize'],
                label=styles[q0]['label']
            )
    plt.axvline(x=critical_alpha, color='black', linestyle='--', linewidth=2)
    plt.title('(b) 攻击成功率', fontsize=13, fontweight='bold')
    plt.xlabel('攻击反馈速率α', fontsize=12)
    plt.ylabel('成功率', fontsize=12)
    plt.xlim(0.04, 0.41)
    plt.ylim(0.0, 0.6)  # 成功率实验观测范围
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # -------------------------- 子图(c)：防御者收益（修复核心） --------------------------
    plt.subplot(2, 2, 3)
    for q0, data in avg_data.items():
        # 强制绘制：只要params非空，即使收益值小也显示（排除维度不匹配问题）
        if len(data['params']) > 0 and len(data['defender_payoff']) == len(data['params']):
            # 打印调试信息（确认数据非空，可删除）
            print(f"q0={q0} 防御者收益数据: {[round(p, 4) for p in data['defender_payoff'][:3]]}")
            plt.plot(
                data['params'], data['defender_payoff'],
                color=styles[q0]['color'], marker=styles[q0]['marker'],
                linewidth=styles[q0]['linewidth'], markersize=styles[q0]['markersize'],
                label=styles[q0]['label']
            )
    plt.axvline(x=critical_alpha, color='black', linestyle='--', linewidth=2)
    plt.title('(c) 防御者收益', fontsize=13, fontweight='bold')
    plt.xlabel('攻击反馈速率α', fontsize=12)
    plt.ylabel('Defender Payoff', fontsize=12)  # 明确纵坐标标签
    plt.xlim(0.04, 0.41)
    plt.ylim(15, 35)  # 扩大范围：避免收益值小被截断（核心修复点）
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # -------------------------- 子图(d)：攻击者收益（修复核心） --------------------------
    plt.subplot(2, 2, 4)
    for q0, data in avg_data.items():
        if len(data['params']) > 0 and len(data['attacker_payoff']) == len(data['params']):
            # 打印调试信息（确认数据非空，可删除）
            print(f"q0={q0} 攻击者收益数据: {[round(p, 4) for p in data['attacker_payoff'][:3]]}")
            plt.plot(
                data['params'], data['attacker_payoff'],
                color=styles[q0]['color'], marker=styles[q0]['marker'],
                linewidth=styles[q0]['linewidth'], markersize=styles[q0]['markersize'],
                label=styles[q0]['label']
            )
    plt.axvline(x=critical_alpha, color='black', linestyle='--', linewidth=2)
    plt.title('(d) 攻击者收益', fontsize=13, fontweight='bold')
    plt.xlabel('攻击反馈速率α', fontsize=12)
    plt.ylabel('Attacker Payoff', fontsize=12)  # 明确纵坐标标签
    plt.xlim(0.04, 0.41)
    plt.ylim(-6, 0)  # 扩大范围：匹配防御者收益趋势，确保曲线可见
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # 调整子图间距：避免标签重叠（关键：2行2列布局更紧凑）
    plt.tight_layout(pad=3.0)  # pad参数增加子图间距离
    # 保存高清图片：避免显示时压缩导致曲线模糊
    plt.savefig('figure6_no_e.png', dpi=300, bbox_inches='tight')
    plt.show()


# 同步修复DataRecorder：确保收益数据与params维度一致（关键辅助）
class DataRecorder:
    def __init__(self):
        self.records = {
            'alphaA': [],
            'coop_rate': [],
            'attack_success_rate': [],
            'defender_payoff': [],
            'attacker_payoff': []  # 移除q_history，无需子图e
        }

    def record_param(self, alphaA, coop, attack, def_payoff, att_payoff):
        """仅记录子图a-d所需数据，避免q_history干扰"""
        self.records['alphaA'].append(alphaA)
        self.records['coop_rate'].append(coop)
        self.records['attack_success_rate'].append(attack)
        self.records['defender_payoff'].append(def_payoff)
        self.records['attacker_payoff'].append(att_payoff)

    def get_averaged_data(self):
        """按alphaA分组计算平均值，强制所有指标维度一致"""
        import numpy as np
        # 1. 获取实验中实际存在的alphaA（避免空值）
        unique_alphaA = sorted([a for a in set(self.records['alphaA']) if a is not None])
        avg_data = {
            'params': unique_alphaA,
            'coop_rate': [],
            'attack_success_rate': [],
            'defender_payoff': [],
            'attacker_payoff': []
        }

        # 2. 按每个alphaA计算平均值（确保无空值）
        for alpha in unique_alphaA:
            # 浮点数容错：匹配当前alpha的所有记录
            indices = [i for i, a in enumerate(self.records['alphaA']) if abs(a - alpha) < 1e-10]
            if not indices:
                # 无数据时填充微小值（避免0值被误判为空）
                avg_data['coop_rate'].append(0.01)
                avg_data['attack_success_rate'].append(0.01)
                avg_data['defender_payoff'].append(0.01)
                avg_data['attacker_payoff'].append(0.01)
                continue

            # 计算平均值：确保收益数据有效
            avg_coop = np.mean([self.records['coop_rate'][i] for i in indices])
            avg_attack = np.mean([self.records['attack_success_rate'][i] for i in indices])
            avg_def = np.mean([self.records['defender_payoff'][i] for i in indices])
            avg_att = np.mean([self.records['attacker_payoff'][i] for i in indices])

            # 强制添加数据：确保每个alphaA对应一条记录
            avg_data['coop_rate'].append(avg_coop)
            avg_data['attack_success_rate'].append(avg_attack)
            avg_data['defender_payoff'].append(avg_def)
            avg_data['attacker_payoff'].append(avg_att)

        return avg_data



def run_exp2_fig6():
    """实验2入口函数：带反馈机制的仿真实验"""
    print("=== Experiment 2：with feedback ===")
    print("Evolutionary dynamics driven by attacker feedback")
    random.seed(42)  # 固定随机种子，保证可复现性
    run_figure6_experiment()