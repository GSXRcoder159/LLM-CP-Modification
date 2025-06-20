
from cpmpy import *
import json

# Parameters
n = 20
m1, m2 = 3, 5

# Decision variables
steps = intvar(0, m2, shape=n)  # steps taken at each move
# New decision variable: longest move size
longest_move = intvar(m1, m2)

# Model setup
m = Model()

# Constraint: link longest_move to steps
m += [longest_move >= steps[i] for i in range(n)]            # ensures longest_move >= every step
m += sum(longest_move == steps[i] for i in range(n)) >= 1     # ensures at least one step equals longest_move

# Constraint: the sum of steps should equal the total number of stairs
m += sum(steps) == n

# Constraint: the number of steps taken at each move should be between m1 and m2 or 0
m += [(steps[i] >= m1) | (steps[i] == 0) for i in range(n)]
m += [steps[i] <= m2 for i in range(n)]

# Trailing zeros: If a step is 0, then all the following steps should be 0
for i in range(1, n):
    m += (steps[i - 1] == 0).implies(all(steps[j] == 0 for j in range(i, n)))

# Solve the model and print the results including longest_move
if m.solve():
    solution = {
        "steps": [steps[i].value() for i in range(n) if steps[i].value() != 0],
        "longest_move": longest_move.value()
    }
    print(json.dumps(solution))
