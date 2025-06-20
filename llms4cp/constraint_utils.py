import cpmpy
from cpmpy import Model
from llms4cp.util import run_code
from llms4cp.dataset_classes import CpmpyModelWithVars

def safe_models_equivalent(model1_str, model2_str):
    """
    A safer implementation for checking if two CPMPy models are equivalent.
    Uses a different approach that avoids modifying constraint args.
    """
    # Clean up the models
    if '```python' in model1_str:
        model1_str = model1_str.split('```python')[1].split('```')[0].strip()
    if '```python' in model2_str:
        model2_str = model2_str.split('```python')[1].split('```')[0].strip()
    
    # If models are exactly the same textually, they are equivalent
    if model1_str.strip() == model2_str.strip():
        return True
    
    # Create a new testing function that avoids the problematic constraint manipulation
    test_code = f"""
from cpmpy import *
import sys

# Model 1
def create_model1():
    {model1_str.replace("# Solve the model", "# Skip solving").replace("m.solve()", "# m.solve()")}
    return m

# Model 2
def create_model2():
    {model2_str.replace("# Solve the model", "# Skip solving").replace("m.solve()", "# m.solve()")}
    return m

# Test if models have the same solution
def check_equivalence():
    try:
        m1 = create_model1()
        m2 = create_model2()
        
        # Try to solve both models
        sol1 = m1.solve()
        if not sol1:
            print("Model 1 is UNSAT")
            # Check if model 2 is also UNSAT
            sol2 = m2.solve()
            return not sol2  # Both UNSAT = equivalent
        
        # Extract solutions from model 1
        m1_vars = [var for var in m1.variables() if hasattr(var, 'value')]
        m1_solution = [var.value() for var in m1_vars]
        
        # Check if solution from model 1 satisfies model 2
        # Create a model that forces model 2 to have the same solution
        m2_check = Model()
        m2_vars = [var for var in m2.variables() if hasattr(var, 'value')]
        
        if len(m1_vars) != len(m2_vars):
            print(f"Different number of variables: {{len(m1_vars)}} vs {{len(m2_vars)}}") 
            return False
        
        # Add constraints to force model 2 vars to match model 1 solution
        for i, var in enumerate(m2_vars):
            if i < len(m1_solution):
                m2_check += (var == m1_solution[i])
        
        # Add all original model 2 constraints
        for c in m2.constraints:
            m2_check += c
        
        # If m2_check is satisfiable, model 1's solution works in model 2
        sol_check = m2_check.solve()
        if not sol_check:
            print("Model 1 solution doesn't satisfy Model 2")
            return False
        
        # Now check the reverse: if model 2's solution satisfies model 1
        m2.solve()  # Get a fresh solution for model 2
        m2_solution = [var.value() for var in m2_vars]
        
        m1_check = Model()
        for i, var in enumerate(m1_vars):
            if i < len(m2_solution):
                m1_check += (var == m2_solution[i])
        
        for c in m1.constraints:
            m1_check += c
        
        reverse_check = m1_check.solve()
        if not reverse_check:
            print("Model 2 solution doesn't satisfy Model 1")
            return False
        
        return True  # Both models accept each other's solutions
        
    except Exception as e:
        print(f"Error in equivalence check: {{e}}")
        return False

# Run the check
result = check_equivalence()
print(f"RESULT:{{result}}")
"""
    
    # Run the equivalence check
    output = run_code(test_code)
    
    if "RESULT:True" in output:
        return True
    else:
        # Print the output for debugging
        print(f"Equivalence check output: {output}")
        return False

def safe_constraint_evaluation(model1_str, model2_str):
    """
    A safer implementation for evaluating constraints between two models.
    Returns (wrong_constraints, total_constraints)
    """
    # If models are exactly the same textually, all constraints are correct
    if model1_str.strip() == model2_str.strip():
        total_constraints = len([line for line in model2_str.split("\n") if "+=" in line and line.strip().startswith("m")])
        return 0, total_constraints
    
    # Extract constraints
    def extract_constraints(model_str):
        return [line.strip() for line in model_str.split("\n") if "+=" in line and line.strip().startswith("m")]
    
    gen_constraints = extract_constraints(model1_str)
    gt_constraints = extract_constraints(model2_str)
    
    if not gt_constraints:
        return 0, 0
    
    # For a simpler implementation, we'll use text similarity as a proxy
    # In a real implementation, you'd want to check semantic equivalence
    similar_constraints = 0
    for gt_constraint in gt_constraints:
        best_similarity = 0
        for gen_constraint in gen_constraints:
            # Simple similarity: normalize and compare
            gt_norm = gt_constraint.replace(" ", "").lower()
            gen_norm = gen_constraint.replace(" ", "").lower()
            if gt_norm == gen_norm:
                similar_constraints += 1
                break
    
    wrong_constraints = len(gt_constraints) - similar_constraints
    return wrong_constraints, len(gt_constraints)