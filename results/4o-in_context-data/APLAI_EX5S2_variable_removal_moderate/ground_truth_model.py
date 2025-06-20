
from cpmpy import *
import json

# Parameters
n = 20
m1 = 3    # m2 removed, we use literal 5 below

# Decision variables
steps = intvar(0, 5, shape=n)  # replaced m2 with literal 5

# Model setup
m = Model()

# Constraint: the sum of steps should equal the total number of stairs
m += sum(steps) == n

# Constraint: the number of steps taken at each move should be between m1 and 5 or 0
m += [(steps[i] >= m1) | (steps[i] == 0) for i in range(n)]
m += [steps[i] <= 5 for i in range(n)]

# Trailing zeros: If a step is 0, then all the following steps should be 0
for i in range(1, n):
    m += (steps[i - 1] == 0).implies(all(steps[j] == 0 for j in range(i, n)))

# Solve the model and print the results in the required format
if m.solve():
    solution = {"steps": [steps[i].value() for i in range(n) if steps[i].value() != 0]}
    print(json.dumps(solution))
