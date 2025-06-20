from cpmpy import *
import json

# Parameters
n = 20  # total number of steps in the stair
m1, m2 = 3, 5  # number of steps that can be taken at a time

# Decision variables
steps = intvar(m1, m2, shape=n)  # steps taken at each move, cannot be 0

# Model setup
m = Model()

# Constraint: the sum of steps should equal the total number of stairs
m += sum(steps) == n

# Constraint: the number of steps taken at each move should be between m1 and m2
m += [steps[i] >= m1 for i in range(n)]
m += [steps[i] <= m2 for i in range(n)]

# Trailing zeros: If a step is 0, then all the following steps should be 0
# This constraint is no longer needed since steps cannot be 0

# Solve the model and print the results in the required format
if m.solve():
    solution = {"steps": [steps[i].value() for i in range(n)]}
    print(json.dumps(solution))