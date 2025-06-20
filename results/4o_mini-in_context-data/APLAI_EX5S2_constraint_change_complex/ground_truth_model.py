
from cpmpy import *
import json

# Parameters
n = 20
m1, m2 = 3, 5

# Decision variables
steps = intvar(0, m2, shape=n)

# Model setup
m = Model()

# Sum constraint unchanged
m += sum(steps) == n
# Domain constraints unchanged
m += [(steps[i] >= m1) | (steps[i] == 0) for i in range(n)]
m += [steps[i] <= m2 for i in range(n)]

# New constraint: forbid two zeros in a row
for i in range(1, n):
    m += ~( (steps[i-1] == 0) & (steps[i] == 0) )  # no two consecutive zeros

# Solve and output
if m.solve():
    solution = {"steps": [s.value() for s in steps if s.value() != 0]}
    print(json.dumps(solution))
