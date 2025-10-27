# experiments/exp3_network_effect.py
from core.simulation import CyberSecuritySimulation
from core.recorder import DefaultDataRecorder


class NetworkEffectSimulation(CyberSecuritySimulation):
    """
    网络效应实验仿真类
    """

    def __init__(self, topology):
        super().__init__(alphaA=0.0, r=6.0, topology=topology)

    def run(self, recorder):
        """
        执行标准仿真流程
        """
        self.run_standard_simulation(recorder)


def run_exp3():
    topologies = ['lattice', 'smallworld', 'scalefree', 'random']
    for topo in topologies:
        print(f"Running topology: {topo}")
        sim = NetworkEffectSimulation(topo)
        rec = DefaultDataRecorder()
        sim.run(rec)
        rec.plot()

if __name__ == "__main__":
    run_exp3()
