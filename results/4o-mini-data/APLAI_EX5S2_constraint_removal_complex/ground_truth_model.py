
from cpmpy import *
import json

# Parameters
n = 20  # total number of steps in the stair
m1, m2 = 3, 5  # number of steps that can be taken at a time

# Decision variables
steps = intvar(0, m2, shape=n) # steps taken at each move

# Model setup
m = Model()

# Constraint: the sum of steps should equal the total number of stairs
m += sum(steps) == n

# (Removed lower-bound and trailing zeros constraints)
# Retain only the upper-bound constraint
m += [steps[i] <= m2 for i in range(n)]

# Solve the model and print the results in the required format
if m.solve():
    solution = {"steps": [steps[i].value() for i in range(n) if steps[i].value() != 0]}
    print(json.dumps(solution))
