# experiments/exp1_no_feedback.py
from core.simulation import CyberSecuritySimulation
from core.recorder import DataRecorder


class NoFeedbackSimulation(CyberSecuritySimulation):
    """
    无反馈机制实验仿真类
    --------------------
    实验条件：alphaA=0.0（无攻击者反馈），r=4.0，拓扑为lattice
    """

    def __init__(self):
        super().__init__(alphaA=0.0, r=4.0, topology='lattice')

    def run(self, recorder):
        """
        执行标准仿真流程
        """
        self.run_standard_simulation(recorder)


def run_exp1():
    sim = NoFeedbackSimulation()
    rec = DataRecorder()
    sim.run(rec)
    rec.plot()

if __name__ == "__main__":
    run_exp1()
