
from cpmpy import *
import json

# Parameters
n = 20
m1, m2 = 3, 5

# Decision variables
steps = intvar(0, m2, shape=n)    # steps taken at each move
# New decision variables: counts of specific step sizes
num_3_steps = intvar(0, n)
num_5_steps = intvar(0, n)

# Model setup
m = Model()

# Constraint: link count variables to steps
m += num_3_steps == sum(steps[i] == m1 for i in range(n))  # count of 3-step moves
m += num_5_steps == sum(steps[i] == m2 for i in range(n))  # count of 5-step moves

# Constraint: the sum of steps should equal the total number of stairs
m += sum(steps) == n

# Constraint: the number of steps taken at each move should be between m1 and m2 or 0
m += [(steps[i] >= m1) | (steps[i] == 0) for i in range(n)]
m += [steps[i] <= m2 for i in range(n)]

# Trailing zeros: If a step is 0, then all the following steps should be 0
for i in range(1, n):
    m += (steps[i - 1] == 0).implies(all(steps[j] == 0 for j in range(i, n)))

# Solve the model and print the results including the counts
if m.solve():
    solution = {
        "steps": [steps[i].value() for i in range(n) if steps[i].value() != 0],
        "num_3_steps": num_3_steps.value(),
        "num_5_steps": num_5_steps.value()
    }
    print(json.dumps(solution))
