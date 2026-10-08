import os
import sys
from opendssdirect import dss

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())
from simulator.scenario_generator import generate_scenarios

# 10k already generated
# generate_scenarios(10000, 'data/state_estimation/scenarios_10k.pkl')
generate_scenarios(50000, 'data/state_estimation/scenarios_50k.pkl')
generate_scenarios(100000, 'data/state_estimation/scenarios_100k.pkl')
