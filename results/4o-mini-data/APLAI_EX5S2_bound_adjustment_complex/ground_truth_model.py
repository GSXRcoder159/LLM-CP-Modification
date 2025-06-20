
from cpmpy import *
import json

# Parameters
n = 21  # changed from 20 to 21
m1, m2 = 3, 5

# Decision variables
steps = intvar(0, m2, shape=n)

# Model setup
m = Model()

# Sum constraint updated
m += sum(steps) == n

# Step-size constraints unchanged
m += [(steps[i] >= m1) | (steps[i] == 0) for i in range(n)]
m += [steps[i] <= m2 for i in range(n)]

# Trailing zeros constraint
for i in range(1, n):
    m += (steps[i-1] == 0).implies(all(steps[j] == 0 for j in range(i, n)))

# Solve and output
if m.solve():
    solution = {"steps": [s.value() for s in steps if s.value() != 0]}
    print(json.dumps(solution))
