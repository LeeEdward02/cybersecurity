# exp2_with_feedback.py
import random
import matplotlib.pyplot as plt
from core.simulation import CyberSecuritySimulation
from core.evolution import fermi_update
from core.games import DefenderAttackerGame

class ExtendedFeedbackSimulation(CyberSecuritySimulation):
    """扩展的带反馈机制仿真类，支持图4、图5和图6的实验设计"""
    
    def __init__(self, alphaA=0.2, q0=0.5, r=4.5, N=1600, rounds=4000, K=0.1, record_param_type='r'):
        super().__init__(
            N=N,
            rounds=rounds,
            r=r,
            q0=q0,
            alphaA=alphaA,
            K=K,
            topology='lattice',
            params=None
        )
        self.record_param_type = record_param_type  # 存储参数类型，用于记录时区分r或alphaA
        # 调整防御者初始策略分布
        coop_count = int(N * 0.5)
        for i, d in enumerate(self.defenders):
            d.strategy = 'C' if i < coop_count else 'D'
        
        # 攻防博弈参数
        self.dag = DefenderAttackerGame(
            gamma1=60,
            gamma2=20,
            delta=70,
            d=60,
            c=15
        )

    def run(self, recorder):
        print(f"开始仿真 (alphaA={self.alphaA}, q0={self.q0}, r={self.r})...")
        
        # 预热轮次
        warmup_rounds = 500
        total_rounds = self.rounds + warmup_rounds
        
        for t in range(total_rounds):
            # 1. 空间公共物品博弈
            for defender in self.defenders:
                group = [defender] + defender.neighbors
                self.pgg.play(group)
            
            # 2. 攻防博弈
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
            
            # 3. 攻击者反馈更新
            local_success = attack_success / total_attacks if total_attacks > 0 else 0
            global_success = local_success
            
            adaptive_alpha = self.alphaA * (0.5 + min(0.5, t / 1000))
            self.attacker.q += adaptive_alpha * (local_success - global_success)
            self.attacker.q = max(0.01, min(0.99, self.attacker.q))
            
            # 4. 防御者策略更新
            for d in self.defenders:
                if d.neighbors:
                    neighbor = random.choice(d.neighbors)
                    dynamic_K = self.K * (5 / self.r) if self.r > 0 else self.K
                    fermi_update(d, neighbor, dynamic_K)
            
            # 5. 数据记录
            coop_rate = sum(d.strategy == 'C' for d in self.defenders) / len(self.defenders)
            attack_success_rate = attack_success / total_attacks if total_attacks > 0 else 0
            
            if t >= warmup_rounds:
                avg_def_payoff = defender_total_payoff / len(self.defenders)
                avg_att_payoff = attacker_total_payoff / len(self.defenders)
                # 记录时包含当前实验参数（图5用r，图6用alphaA）
                # 根据record_param_type选择记录的参数
                if self.record_param_type == 'r':
                    param = self.r  # 图5实验：记录r作为参数
                else:
                    param = self.alphaA  # 图6实验：记录alphaA作为参数

                recorder.record_param(
                    param=self.r,  # 图5实验用r作为参数
                    coop=coop_rate,
                    attack=attack_success_rate,
                    q=self.attacker.q,
                    def_payoff=avg_def_payoff,
                    att_payoff=avg_att_payoff
                )
            
            # 6. 重置收益
            for d in self.defenders:
                d.reset_payoff()
            
            # 进度显示
            if (t + 1) % 500 == 0:
                print(f"  第 {t+1}/{total_rounds} 轮: 合作率={coop_rate:.3f}, "
                    f"攻击成功率={attack_success_rate:.3f}, 攻击概率={self.attacker.q:.3f}")


def run_figure4_experiment():
    """运行图4的实验：不同alpha值下系统的时间演化"""
    print("=== 图4实验：不同alpha值下系统的时间演化 ===")
    alpha_values = [0.15, 0.16, 0.18, 0.20]  # 不同攻击反馈速率
    q0 = 0.5  # 固定初始攻击概率
    r = 4.5   # 固定增强因子
    rounds = 4000  # 4000轮次
    
    # 为每个alpha创建记录器
    recorders = {alpha: DataRecorder() for alpha in alpha_values}
    
    for alphaA in alpha_values:
        print(f"\n--- 测试 alphaA={alphaA} ---")
        sim = ExtendedFeedbackSimulation(
            alphaA=alphaA,
            q0=q0,
            r=r,
            rounds=rounds
        )
        sim.run(recorders[alphaA])
    
    # 绘制图4的四个子图
    plt.figure(figsize=(15, 16))
    time_steps = list(range(rounds))
    
    # (a) 防御者合作水平
    plt.subplot(2, 2, 1)
    for alphaA, recorder in recorders.items():
        plt.plot(time_steps, recorder.records['coop_rate'], label=f'α={alphaA}')
    plt.title('(a) 防御者合作水平')
    plt.xlabel('轮次')
    plt.ylabel('合作率')
    plt.legend()
    
    # (b) 攻击成功率
    plt.subplot(2, 2, 2)
    for alphaA, recorder in recorders.items():
        plt.plot(time_steps, recorder.records['attack_success_rate'], label=f'α={alphaA}')
    plt.title('(b) 攻击成功率')
    plt.xlabel('轮次')
    plt.ylabel('成功率')
    plt.legend()
    
    # (c) 防御者收益
    plt.subplot(2, 2, 3)
    for alphaA, recorder in recorders.items():
        plt.plot(time_steps, recorder.records['defender_payoff'], label=f'α={alphaA}')
    plt.title('(c) 防御者收益')
    plt.xlabel('轮次')
    plt.ylabel('平均收益')
    plt.legend()
    
    # (d) 攻击者收益
    plt.subplot(2, 2, 4)
    for alphaA, recorder in recorders.items():
        plt.plot(time_steps, recorder.records['attacker_payoff'], label=f'α={alphaA}')
    plt.title('(d) 攻击者收益')
    plt.xlabel('轮次')
    plt.ylabel('平均收益')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def run_figure5_experiment():
    """运行图5的实验：不同alpha值下系统随r的变化"""
    print("=== 图5实验：不同alpha值下系统随r的变化 ===")
    alpha_values = [0.05, 0.1, 0.15]  # 不同攻击反馈速率
    r_values = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]  # r值范围
    q0 = 0.5  # 固定初始攻击概率
    
    # 为每个alpha创建记录器
    recorders = {alpha: DataRecorder() for alpha in alpha_values}
    
    for r in r_values:
        print(f"\n--- 测试 r={r} ---")
        for alphaA in alpha_values:
            sim = ExtendedFeedbackSimulation(
                alphaA=alphaA,
                q0=q0,
                r=r,
                rounds=1000,
                record_param_type='r'   # 每个r值的稳定期仿真轮数
            )
            sim.run(recorders[alphaA])
    
    # 绘制图5的四个子图
    plt.figure(figsize=(15, 16))
    
    # 获取平均数据
    avg_data = {alpha: recorder.get_averaged_data() for alpha, recorder in recorders.items()}
    
    # (a) 防御者合作水平
    plt.subplot(2, 2, 1)
    for alphaA, data in avg_data.items():
        plt.plot(data['params'], data['coop_rate'], marker='o', label=f'αₐ={alphaA}')
    plt.axvline(x=3.5, color='gray', linestyle='--', label='临界r值')
    plt.title('(a) 防御者合作水平')
    plt.xlabel('增强因子r')
    plt.ylabel('合作率')
    plt.legend()
    
    # (b) 攻击成功率
    plt.subplot(2, 2, 2)
    for alphaA, data in avg_data.items():
        plt.plot(data['params'], data['attack_success_rate'], marker='o', label=f'α={alphaA}')
    plt.axvline(x=3.5, color='gray', linestyle='--')
    plt.title('(b) 攻击成功率')
    plt.xlabel('增强因子r')
    plt.ylabel('成功率')
    plt.legend()
    
    # (c) 防御者收益
    plt.subplot(2, 2, 3)
    for alphaA, data in avg_data.items():
        plt.plot(data['params'], data['defender_payoff'], marker='o', label=f'α={alphaA}')
    plt.axvline(x=3.5, color='gray', linestyle='--')
    plt.title('(c) 防御者收益')
    plt.xlabel('增强因子r')
    plt.ylabel('平均收益')
    plt.legend()
    
    # (d) 攻击者收益
    plt.subplot(2, 2, 4)
    for alphaA, data in avg_data.items():
        plt.plot(data['params'], data['attacker_payoff'], marker='o', label=f'α={alphaA}')
    plt.axvline(x=3.5, color='gray', linestyle='--')
    plt.title('(d) 攻击者收益')
    plt.xlabel('增强因子r')
    plt.ylabel('平均收益')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


class DataRecorder:
    """扩展的数据记录类，支持按参数分组记录"""
    def __init__(self):
        self.records = {
            'params': [],  # 存储实验参数（r或alphaA）
            'coop_rate': [],
            'attack_success_rate': [],
            'q_history': [],
            'defender_payoff': [],
            'attacker_payoff': []
        }

    def record_param(self, param, coop, attack, q, def_payoff, att_payoff):
        """按参数记录一组数据"""
        self.records['params'].append(param)
        self.records['coop_rate'].append(coop)
        self.records['attack_success_rate'].append(attack)
        self.records['q_history'].append(q)
        self.records['defender_payoff'].append(def_payoff)
        self.records['attacker_payoff'].append(att_payoff)

    def get_averaged_data(self, target_params=None):
        """按参数分组并计算平均值"""
        import numpy as np
        if target_params is not None:
            unique_params = target_params
        else:
            unique_params = sorted(list(set(self.records['params'])))
        avg_data = {p: {} for p in unique_params}
        
        for p in unique_params:
            # 收集该参数下的所有记录
            indices = [i for i, param in enumerate(self.records['params']) if param == p]
            for key in ['coop_rate', 'attack_success_rate', 'q_history', 
                       'defender_payoff', 'attacker_payoff']:
                values = [self.records[key][i] for i in indices]
                avg_data[p][key] = np.mean(values)  # 计算平均值
        
        # 转换为绘图所需的列表格式
        params = sorted(avg_data.keys())
        return {
            'params': params,
            'coop_rate': [avg_data[p]['coop_rate'] for p in params],
            'attack_success_rate': [avg_data[p]['attack_success_rate'] for p in params],
            'q_history': [avg_data[p]['q_history'] for p in params],
            'defender_payoff': [avg_data[p]['defender_payoff'] for p in params],
            'attacker_payoff': [avg_data[p]['attacker_payoff'] for p in params]
        }


def run_exp2():
    """实验2入口函数：带反馈机制的仿真实验"""
    print("=== 实验2：带反馈机制 ===")
    print("研究攻击者反馈机制对系统演化的影响")
    random.seed(42)  # 固定随机种子，保证可复现性
    run_figure4_experiment()
    run_figure5_experiment()