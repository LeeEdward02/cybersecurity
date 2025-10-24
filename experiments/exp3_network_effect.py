# experiments/exp3_network_effect.py
from core.simulation import CyberSecuritySimulation
from core.recorder import DataRecorder

def run_exp3():
    topologies = ['lattice', 'smallworld', 'scalefree', 'random']
    for topo in topologies:
        print(f"Running topology: {topo}")
        sim = CyberSecuritySimulation(alphaA=0.0, r=6.0, topology=topo)
        rec = DataRecorder()
        sim.run(rec)
        rec.plot()

if __name__ == "__main__":
    run_exp3()
