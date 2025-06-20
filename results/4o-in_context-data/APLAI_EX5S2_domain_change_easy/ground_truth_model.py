
from cpmpy import *
import json

# Parameters
n = 20
m1, m2 = 3, 5

# Decision variables
steps = intvar(0, m2+1, shape=n)  # domain changed to include m2+1

# Model setup
m = Model()

m += sum(steps) == n

m += [(steps[i] >= m1) | (steps[i] == 0) for i in range(n)]
m += [steps[i] <= m2+1 for i in range(n)]
for i in range(1, n):
    m += (steps[i-1] == 0).implies(all(steps[j] == 0 for j in range(i, n)))

if m.solve():
    solution = {"steps": [s.value() for s in steps if s.value() != 0]}
    print(json.dumps(solution))
