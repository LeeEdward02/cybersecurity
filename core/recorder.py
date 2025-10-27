"""
recorder.py
------------
负责记录仿真过程中的指标（合作率、攻击成功率、攻击概率等）
并提供可视化功能。
"""

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


class DataRecorder(ABC):
    """
    数据记录与可视化类
    -----------------
    记录每轮的主要仿真指标并可绘制结果曲线。
    """

    def __init__(self):
        """
        初始化记录结构。
        """
        self.records = {'coop_rate': [],
                        'attack_success_rate': [],
                        'q_history': [],
                        'defender_payoff': [],
                        }

    @abstractmethod
    def record(self, coop, attack, q, payoff):
        """
        记录当前轮的关键指标。

        Args:
            coop (float): 合作率（合作节点数量 / 总节点数）
            attack (float): 攻击成功率
            q (float): 当前攻击概率 q(t)
            payoff (float): 防御者总收益
        Returns:
            None
        """
        pass

    @abstractmethod
    def plot(self):
        """
        绘制实验结果曲线。
        Returns:
            None（直接展示图像）
        """
        pass


class DefaultDataRecorder(DataRecorder):
    """
    默认数据记录实现
    -----------------
    提供基础的数据记录功能，用于向后兼容。
    """

    def record(self, coop, attack, q, payoff):
        """
        记录当前轮的关键指标。

        Args:
            coop (float): 合作率（合作节点数量 / 总节点数）
            attack (float): 攻击成功率
            q (float): 当前攻击概率 q(t)
            payoff (float): 防御者总收益
        Returns:
            None
        """
        self.records['coop_rate'].append(coop)
        self.records['attack_success_rate'].append(attack)
        self.records['q_history'].append(q)
        self.records['defender_payoff'].append(payoff)

    def plot(self):
        """
        绘制实验结果曲线。
        Returns:
            None（直接展示图像）
        """
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.records['coop_rate'], label='合作率')
        plt.plot(self.records['attack_success_rate'], label='攻击成功率')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(self.records['q_history'], label='攻击概率 q(t)')
        plt.legend()
        plt.show()
