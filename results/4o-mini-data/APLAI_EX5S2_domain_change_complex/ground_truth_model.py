
from cpmpy import *
import json

# Parameters
n = 20
m1, m2 = 3, 5

# Decision variables
steps = intvar(m1, m2, shape=n)  # domain changed to exclude 0

# Model setup
m = Model()

# Sum constraint
m += sum(steps) == n

# Domain constraints simplified (0 no longer possible)
m += [steps[i] >= m1 for i in range(n)]
m += [steps[i] <= m2 for i in range(n)]

# Solve and output
if m.solve():
    solution = {"steps": [s.value() for s in steps]}
    print(json.dumps(solution))
