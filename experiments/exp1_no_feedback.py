# experiments/exp1_no_feedback.py
from core.simulation import CyberSecuritySimulation
from core.recorder import DataRecorder

def run_exp1():
    sim = CyberSecuritySimulation(alphaA=0.0, r=4.0, topology='lattice')
    rec = DataRecorder()
    sim.run(rec)
    rec.plot()

if __name__ == "__main__":
    run_exp1()
