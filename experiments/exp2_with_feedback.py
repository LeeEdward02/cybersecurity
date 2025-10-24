# experiments/exp2_with_feedback.py
from core.simulation import CyberSecuritySimulation
from core.recorder import DataRecorder

def run_exp2():
    sim = CyberSecuritySimulation(alphaA=0.2, q0=0.5, r=4.5, topology='lattice')
    rec = DataRecorder()
    sim.run(rec)
    rec.plot()

if __name__ == "__main__":
    run_exp2()
