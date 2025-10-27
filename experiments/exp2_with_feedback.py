# experiments/exp2_with_feedback.py
from core.simulation import CyberSecuritySimulation
from core.recorder import DefaultDataRecorder


class WithFeedbackSimulation(CyberSecuritySimulation):
    """
    带反馈机制实验仿真类
    """

    def __init__(self):
        super().__init__(alphaA=0.2, q0=0.5, r=4.5, topology='lattice')

    def run(self, recorder):
        """
        执行标准仿真流程
        """
        self.run_standard_simulation(recorder)


def run_exp2():
    sim = WithFeedbackSimulation()
    rec = DefaultDataRecorder()
    sim.run(rec)
    rec.plot()

if __name__ == "__main__":
    run_exp2()
