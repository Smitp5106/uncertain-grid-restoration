import os
import sys
from opendssdirect import dss

# Change working directory so relative path works
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

dss("Clear")

dss('Redirect "dss/ieee33/Master.dss"')

dss.Solution.Solve()

print("Solved:", dss.Solution.Converged())

print("\nBuses:")
for bus in dss.Circuit.AllBusNames():
    print(bus)

print("\nVoltages:")
print(dss.Circuit.AllBusMagPu())
