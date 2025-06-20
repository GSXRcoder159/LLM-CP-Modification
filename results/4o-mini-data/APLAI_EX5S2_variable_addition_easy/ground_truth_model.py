
from cpmpy import *
import json

# Parameters
n = 20
m1, m2 = 3, 5

# Decision variables
steps = intvar(0, m2, shape=n)  # steps taken at each move
# New decision variable: count of non-zero moves
total_moves = intvar(0, n)

# Model setup
m = Model()

# Constraint: link total_moves to number of non-zero steps
m += total_moves == sum(steps[i] != 0 for i in range(n))  # added variable linkage

# Constraint: the sum of steps should equal the total number of stairs
m += sum(steps) == n

# Constraint: the number of steps taken at each move should be between m1 and m2 or 0
m += [(steps[i] >= m1) | (steps[i] == 0) for i in range(n)]
m += [steps[i] <= m2 for i in range(n)]

# Trailing zeros: If a step is 0, then all the following steps should be 0
for i in range(1, n):
    m += (steps[i - 1] == 0).implies(all(steps[j] == 0 for j in range(i, n)))

# Solve the model and print the results including total_moves
if m.solve():
    solution = {
        "steps": [steps[i].value() for i in range(n) if steps[i].value() != 0],
        "total_moves": total_moves.value()
    }
    print(json.dumps(solution))
