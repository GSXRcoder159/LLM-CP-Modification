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

# Constraint: the number of steps taken at each move should be between m1 and m2 or 0
m += [(steps[i] >= m1) | (steps[i] == 0) for i in range(n)]
m += [steps[i] <= m2 for i in range(n)]

# Trailing zeros: If a step is 0, then all the following steps should be 0
for i in range(1, n):
    m += (steps[i - 1] == 0).implies(all(steps[j] == 0 for j in range(i, n)))

# Objective: maximize 10x(number of largest step-size moves) - (number of moves)
num_largest_steps = sum(steps[i] == m2 for i in range(n))
num_moves = sum(steps[i] != 0 for i in range(n))
m.maximize(10 * num_largest_steps - num_moves)

# Solve the model and print the results in the required format
if m.solve():
    solution = {"steps": [steps[i].value() for i in range(n) if steps[i].value() != 0]}
    print(json.dumps(solution))