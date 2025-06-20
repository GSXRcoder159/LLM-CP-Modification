
from cpmpy import *
import json

# Parameters
n = 20  # total number of steps in the stair
m1, m2 = 3, 5  # number of steps that can be taken at a time

# Decision variables
steps = intvar(0, m2, shape=n-1) # reduced to n-1 moves

# Model setup
m = Model()

# Constraint: the sum of steps should equal the total number of stairs
m += sum(steps) == n

# Constraint: the number of steps taken at each move should be between m1 and m2 or 0
m += [(steps[i] >= m1) | (steps[i] == 0) for i in range(n-1)]
m += [steps[i] <= m2 for i in range(n-1)]
# Trailing zeros: once a 0 is encountered, all following must be 0
for i in range(1, n-1):
    m += (steps[i - 1] == 0).implies(all(steps[j] == 0 for j in range(i, n-1)))
# Solve the model and print the results
if m.solve():
    solution = {"steps": [steps[i].value() for i in range(n-1) if steps[i].value() != 0]}
    print(json.dumps(solution))
