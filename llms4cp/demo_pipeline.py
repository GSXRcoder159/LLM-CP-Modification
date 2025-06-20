import json
import os
import re
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython.display import display, HTML, Markdown
from io import BytesIO
import base64
import difflib

from llms4cp.call_llms import call_llm
from llms4cp.data_reader import read_aplai_data
from llms4cp.in_context_config import SYSTEM_MESSAGE_APLAI_SOLVE, SYSTEM_MESSAGE_PSEUDO_APLAI, SYSTEM_MESSAGE_CPMPY_APLAI
from llms4cp.lgps_cpmpy_model_equivalence import check_cpmpy_str_models_equivalence, get_cpmpy_str_per_const_eval
from llms4cp.util import get_examples_for_context, choose_example_selector, run_code, get_number_of_constraints, format_final_executable, execute_and_get_solution, get_var_names_from_exec_model_str

def demo_aplai_pipeline(problem_id=None, method="CPMPY", num_examples=4, example_selector="static"):
    """
    Demonstration pipeline that reveals all internal steps and metrics.
    
    Args:
        problem_id: ID or index of the problem to demonstrate
        method: One of "DIRECT", "CPMPY", "PSEUDO", "NER"
        num_examples: Number of examples for in-context learning
        example_selector: Method for selecting examples
        
    Returns:
        Dictionary with all intermediate steps and metrics
    """
    # Configure and initialize
    os.environ["NUM_EXAMPLES"] = str(num_examples)
    os.environ["EXAMPLES_SELECTOR"] = example_selector
    
    # Load dataset
    all_problems = read_aplai_data()
    if problem_id is None:
        raise ValueError("Problem ID is required.")
    
    if isinstance(problem_id, int):
        problem = all_problems[problem_id]
    else:
        raise NotImplementedError("Only integer problem IDs are supported.")
    
    # Exclude current problem from examples
    dataset_without_current = [p for i, p in enumerate(all_problems) if i != problem_id]
    
    # Track all steps and results
    demo_results = {
        "problem": {
            "id": problem_id,
            "description": problem.description,
            "ground_truth_model": problem.cpmpy_code,
            "question": problem.question,
            "entities": problem.entities_as_str if hasattr(problem, "entities_as_str") else None
        },
        "config": {
            "method": method,
            "num_examples": num_examples,
            "example_selector": example_selector
        },
        "pipeline_steps": [],
        "evaluation_metrics": {},
        "evaluation_details": {}
    }
    
    # Step 1: Example Selection
    example_selector_obj = choose_example_selector(dataset_without_current)
    
    # Prepare input based on method
    input_vars = {"question": problem.question}
    question = problem.question
    
    demo_results["pipeline_steps"].append({
        "step": "Problem Description",
        "details": question
    })
    
    # Add NER if required
    if method == "NER":
        input_vars["ner"] = problem.entities_as_str
        question += '\n' + problem.entities_as_str
        demo_results["pipeline_steps"].append({
            "step": "Named Entity Recognition",
            "details": problem.entities_as_str
        })
    
    # Get examples for context
    examples = get_examples_for_context(example_selector_obj, input_vars, dataset_without_current)
    
    demo_results["pipeline_steps"].append({
        "step": "Selected Examples",
        "details": examples
    })
    
    # Step 2: Generate Pseudo Model (if applicable)
    pseudo_model = None
    if method in ["PSEUDO", "NER"]:
        messages, pseudo_model = call_llm(
            examples,
            question,
            system_message=SYSTEM_MESSAGE_PSEUDO_APLAI,
            method=method,
            step='GEN_PSEUDO'
        )
        input_vars["pseudo_model"] = pseudo_model
        question += '\n' + pseudo_model
        
        demo_results["pipeline_steps"].append({
            "step": "Generated Pseudo Model",
            "details": pseudo_model,
            "prompt": messages
        })

    # Step 3: Generate CPMPy Model
    # messages, cpmpy_model = call_llm(
    #     examples,
    #     question,
    #     system_message=SYSTEM_MESSAGE_CPMPY_APLAI,
    #     method=method,
    #     step='GEN_CPMPY'
    # )
    
    cpmpy_model = f"""
```python
from cpmpy import *
import json

# Decision Variables
a = intvar(0, 9)  # First digit
b = intvar(0, 9)  # Second digit
c = intvar(0, 9)  # Third digit
d = intvar(0, 9)  # Fourth digit

# Constraints
m = Model()

m += AllDifferent([a, b, c, d])  # No two digits are the same
m += (10 * c + d) == 3 * (10 * a + b)  # cd is 3 times ab
m += (10 * d + a) == 2 * (10 * b + c)  # da is 2 times bc
```
"""
    
    # Extract clean code without markdown
    if '```python' in cpmpy_model:
        clean_cpmpy_model = cpmpy_model.split('```python')[1].split('```')[0].strip()
    else:
        clean_cpmpy_model = cpmpy_model.strip()

    # Debugging Only
    # cpmpy_model = problem.cpmpy_code
    # clean_cpmpy_model = cpmpy_model
    messages = "No messages available"
        
    demo_results["pipeline_steps"].append({
        "step": "Generated CPMPy Model",
        "details": cpmpy_model,
        "clean_code": clean_cpmpy_model,
        "prompt": messages
    })
    
    # Step A: Syntactic Correctness Evaluation
    syntax_evaluation_details = evaluate_syntactic_correctness(clean_cpmpy_model)
    demo_results["evaluation_metrics"]["syntactic_correctness"] = {
        "correct": syntax_evaluation_details["correct"],
        "execution_output": syntax_evaluation_details["execution_output"]
    }
    demo_results["evaluation_details"]["syntactic_correctness"] = syntax_evaluation_details
    
    # Step B: Solution-Level Evaluation
    solution_evaluation_details = evaluate_solution_correctness(clean_cpmpy_model, problem.cpmpy_code)
    demo_results["evaluation_metrics"]["solution_evaluation"] = {
        "correct": solution_evaluation_details["direct_match"],
        "ground_truth_solution": solution_evaluation_details["ground_truth_solution"],
        "generated_solution": solution_evaluation_details["generated_solution"]
    }
    
    if "validation_correct" in solution_evaluation_details:
        demo_results["evaluation_metrics"]["solution_validation"] = {
            "correct": solution_evaluation_details["validation_correct"],
            "validation_output": solution_evaluation_details["validation_output"]
        }
    
    demo_results["evaluation_details"]["solution_evaluation"] = solution_evaluation_details
    
    # Step C: Model-Level Equivalence
    model_evaluation_details = evaluate_model_equivalence(clean_cpmpy_model, problem.cpmpy_code)
    demo_results["evaluation_metrics"]["model_equivalence"] = {
        "equivalent": model_evaluation_details["equivalent"]
    }
    demo_results["evaluation_details"]["model_equivalence"] = model_evaluation_details
    
    # Step D: Constraint-Level Evaluation
    constraint_evaluation_details = evaluate_constraints(clean_cpmpy_model, problem.cpmpy_code)
    
    if constraint_evaluation_details["success"]:
        demo_results["evaluation_metrics"]["constraint_evaluation"] = {
            "wrong_constraints": constraint_evaluation_details["wrong_constraints"],
            "total_constraints": constraint_evaluation_details["total_constraints"],
            "constraint_accuracy": constraint_evaluation_details["constraint_accuracy"]
        }
    else:
        demo_results["evaluation_metrics"]["constraint_evaluation"] = {
            "error": constraint_evaluation_details["error_message"]
        }
    
    demo_results["evaluation_details"]["constraint_evaluation"] = constraint_evaluation_details
    
    # Step E: Final Accuracy Calculation
    syntactic_correct = demo_results["evaluation_metrics"]["syntactic_correctness"].get("correct", False)
    solution_correct = demo_results["evaluation_metrics"]["solution_evaluation"].get("correct", False)
    validation_correct = demo_results["evaluation_metrics"].get("solution_validation", {}).get("correct", False)
    model_equivalent = demo_results["evaluation_metrics"]["model_equivalence"].get("equivalent", False)
    
    # Safely get constraint accuracy, handling possible errors
    constraint_accuracy = 0
    if "constraint_evaluation" in demo_results["evaluation_metrics"]:
        constraint_accuracy = demo_results["evaluation_metrics"]["constraint_evaluation"].get("constraint_accuracy", 0)
        if not isinstance(constraint_accuracy, (int, float)):
            constraint_accuracy = 0
    
    demo_results["evaluation_metrics"]["final_accuracy"] = {
        "syntactic_correct": syntactic_correct,
        "solution_correct": solution_correct or validation_correct,
        "model_equivalent": model_equivalent,
        "constraint_accuracy": constraint_accuracy
    }
    
    return demo_results

def evaluate_syntactic_correctness(cpmpy_model):
    """Detailed evaluation of syntactic correctness with line-by-line analysis."""
    result = {
        "correct": False,
        "execution_output": "",
        "line_by_line_analysis": [],
        "success": False,
        "error_message": "",
        "error_line": -1
    }
    
    try:
        execution_output = run_code(cpmpy_model)
        syntax_correct = "Error" not in execution_output
        
        result["correct"] = syntax_correct
        result["execution_output"] = execution_output
        result["success"] = True
        
        # Line-by-line syntax analysis
        lines = cpmpy_model.split('\n')
        for i, line in enumerate(lines):
            line_result = {
                "line_number": i + 1,
                "code": line,
                "status": "valid"
            }
            result["line_by_line_analysis"].append(line_result)
            
        if not syntax_correct and "Error" in execution_output:
            # Try to extract error line number
            error_match = re.search(r"line (\d+)", execution_output)
            if error_match:
                error_line = int(error_match.group(1))
                result["error_line"] = error_line
                
                # Mark the erroneous line
                for line_info in result["line_by_line_analysis"]:
                    if line_info["line_number"] == error_line:
                        line_info["status"] = "error"
                        line_info["error"] = execution_output
                        break
        
    except Exception as e:
        result["correct"] = False
        result["execution_output"] = str(e)
        result["success"] = False
        result["error_message"] = str(e)
        
        # Try to identify the problematic line if possible
        if "line" in str(e):
            error_match = re.search(r"line (\d+)", str(e))
            if error_match:
                error_line = int(error_match.group(1))
                result["error_line"] = error_line
    
    return result

def evaluate_solution_correctness(generated_model, ground_truth_model):
    """Detailed evaluation of solution correctness with comparison breakdown."""
    result = {
        "direct_match": False,
        "ground_truth_solution": None,
        "generated_solution": None,
        "group_by_group_comparison": [],
        "success": False,
        "error_message": ""
    }
    
    try:
        # Get solution from ground truth
        ground_truth_solution = execute_and_get_solution(ground_truth_model)
        result["ground_truth_solution"] = ground_truth_solution
        
        # Get solution from generated model
        generated_solution = execute_and_get_solution(generated_model)
        result["generated_solution"] = generated_solution
        
        # No solution cases
        if ground_truth_solution is None:
            result["direct_match"] = (generated_solution is None)
            result["success"] = True
            return result
            
        if generated_solution is None:
            result["direct_match"] = False
            result["success"] = True
            return result
        
        # Direct solution comparison
        result["direct_match"] = are_solutions_equivalent(ground_truth_solution, generated_solution)
        
        # Group-by-group comparison
        max_groups = max(len(ground_truth_solution), len(generated_solution))
        for i in range(max_groups):
            group_comparison = {
                "group_number": i + 1,
                "ground_truth_group": ground_truth_solution[i] if i < len(ground_truth_solution) else [],
                "generated_group": generated_solution[i] if i < len(generated_solution) else [],
                "match": False
            }
            
            if (i < len(ground_truth_solution) and 
                i < len(generated_solution) and 
                set(ground_truth_solution[i]) == set(generated_solution[i])):
                group_comparison["match"] = True
                
            result["group_by_group_comparison"].append(group_comparison)
        
        # Alternative solution validation (using the final_executable approach)
        if generated_solution:
            validation_code = format_final_executable(ground_truth_model, generated_solution)
            validation_output = run_code(validation_code)
            validation_correct = "Model Solved: True" in validation_output
            
            result["validation_correct"] = validation_correct
            result["validation_output"] = validation_output
        
        result["success"] = True
        
    except Exception as e:
        result["direct_match"] = False
        result["success"] = False
        result["error_message"] = str(e)
    
    return result

def are_solutions_equivalent(sol1, sol2):
    """Check if two solutions are equivalent (same groups, possibly in different order)."""
    if sol1 is None and sol2 is None:
        return True
    if sol1 is None or sol2 is None:
        return False
    if len(sol1) != len(sol2):
        return False
    
    # Convert all groups to sets for comparison
    sol1_sets = [set(group) for group in sol1]
    sol2_sets = [set(group) for group in sol2]
    
    # Check if each set in sol1 appears in sol2
    for group1 in sol1_sets:
        if group1 not in sol2_sets:
            return False
    
    return True

def evaluate_model_equivalence(generated_model, ground_truth_model):
    """Detailed evaluation of model equivalence with constraint comparison."""
    result = {
        "equivalent": False,
        "success": False,
        "error_message": "",
        "details": "Model equivalence checks if the two models have the same solution space. " +
                   "This is done by checking if one model entails the other and vice versa."
    }
    
    try:
        model_equivalent = check_cpmpy_str_models_equivalence(generated_model, ground_truth_model, True)
        result["equivalent"] = model_equivalent
        result["success"] = True
        
        # Add details about what was checked
        if model_equivalent:
            result["details"] += "\n\nThe models are equivalent, meaning they represent the same constraint problem and have the same solution space."
        else:
            result["details"] += "\n\nThe models are not equivalent, meaning they represent different constraint problems or have different solution spaces."
        
    except Exception as e:
        result["equivalent"] = False
        result["success"] = False
        result["error_message"] = str(e)
    
    return result

def evaluate_constraints(generated_model, ground_truth_model):
    """Detailed evaluation of constraints with comparison breakdown."""
    result = {
        "success": False,
        "error_message": "",
        "constraint_accuracy": 0,
        "constraint_comparison": []
    }
    
    try:
        # Get constraints from both models
        gt_constraints = extract_constraints(ground_truth_model)
        gen_constraints = extract_constraints(generated_model)
        
        # Fast path for identical models (e.g., when using ground truth for debugging)
        if generated_model.strip() == ground_truth_model.strip():
            result["wrong_constraints"] = 0
            result["total_constraints"] = len(gt_constraints)
            result["constraint_accuracy"] = 1.0
            result["success"] = True
            
            # Still create a detailed comparison for visualization
            for i, constraint in enumerate(gt_constraints):
                constraint_comparison = {
                    "index": i,
                    "ground_truth_constraint": constraint,
                    "generated_constraint": constraint,
                    "semantically_equivalent": True,
                    "syntactically_similar": True,
                    "similarity_score": 1.0
                }
                result["constraint_comparison"].append(constraint_comparison)
                
            return result
        
        try:
            # Detailed constraint evaluation using CPMPy equivalence
            wrong_constraints, total_constraints = get_cpmpy_str_per_const_eval(generated_model, ground_truth_model, True)
            
            # Calculate constraint accuracy
            constraint_accuracy = (total_constraints - wrong_constraints) / total_constraints if total_constraints > 0 else 0
            
            # Store the results
            result["success"] = True
            result["wrong_constraints"] = wrong_constraints
            result["total_constraints"] = total_constraints
            result["constraint_accuracy"] = constraint_accuracy
        except Exception as inner_e:
            print(f"Error in constraint evaluation: {inner_e}")
            result["error_message"] = f"Error in detailed constraint evaluation: {str(inner_e)}"
            
            # Fall back to simpler comparison for demo purposes
            gt_constraints = extract_constraints(ground_truth_model)
            gen_constraints = extract_constraints(generated_model)
            
            # Simple text-based comparison fallback
            wrong_constraints = 0
            for gt_const in gt_constraints:
                matched = False
                for gen_const in gen_constraints:
                    if gt_const.replace(" ", "") == gen_const.replace(" ", ""):
                        matched = True
                        break
                if not matched:
                    wrong_constraints += 1
            
            result["wrong_constraints"] = wrong_constraints
            result["total_constraints"] = len(gt_constraints)
            result["constraint_accuracy"] = (len(gt_constraints) - wrong_constraints) / len(gt_constraints) if len(gt_constraints) > 0 else 0
        
        # Create a detailed comparison of constraints regardless of whether CPMPy evaluation worked
        max_constraints = max(len(gt_constraints), len(gen_constraints))
        for i in range(max_constraints):
            constraint_comparison = {
                "index": i,
                "ground_truth_constraint": gt_constraints[i] if i < len(gt_constraints) else None,
                "generated_constraint": gen_constraints[i] if i < len(gen_constraints) else None,
                "semantically_equivalent": False,
                "syntactically_similar": False,
                "similarity_score": 0
            }
            
            # If both constraints exist, calculate similarity
            if (constraint_comparison["ground_truth_constraint"] is not None and 
                constraint_comparison["generated_constraint"] is not None):
                
                # Calculate string similarity as a proxy for syntactic similarity
                similarity = calculate_string_similarity(
                    constraint_comparison["ground_truth_constraint"],
                    constraint_comparison["generated_constraint"]
                )
                constraint_comparison["similarity_score"] = similarity
                constraint_comparison["syntactically_similar"] = similarity > 0.7
                
                # If we have CPMPy evaluation results, use them to determine semantic equivalence
                if "wrong_constraints" in result and result["success"]:
                    try:
                        if i < len(gt_constraints) - result["wrong_constraints"]:
                            constraint_comparison["semantically_equivalent"] = True
                    except:
                        pass
            
            result["constraint_comparison"].append(constraint_comparison)
        
    except Exception as e:
        result["success"] = False
        result["error_message"] = str(e)
    
    return result

def extract_constraints(model_code):
    """Extract constraint lines from a model code string."""
    constraints = []
    lines = model_code.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('m +='):
            constraints.append(line)
    return constraints

def calculate_string_similarity(str1, str2):
    """Calculate similarity between two strings using difflib."""
    return difflib.SequenceMatcher(None, str1, str2).ratio()

def visualize_syntactic_correctness(demo_results):
    """Visualize syntactic correctness evaluation with code highlighting."""
    if 'syntactic_correctness' not in demo_results['evaluation_details']:
        return HTML("<p>Syntactic correctness evaluation data not available</p>")
    
    eval_data = demo_results['evaluation_details']['syntactic_correctness']
    
    # Create header with overall result
    header = f"""
    <div style="margin-bottom: 20px;">
        <h3>Syntactic Correctness Evaluation</h3>
        <p><strong>Result:</strong> {"✓ Passed" if eval_data['correct'] else "✗ Failed"}</p>
    </div>
    """
    
    # Create code display with line highlighting
    code_lines = demo_results['pipeline_steps'][-1]['clean_code'].split('\n')
    code_html = '<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; font-family: monospace;">'
    
    for i, line in enumerate(code_lines):
        line_num = i + 1
        background_color = "#ffffff"  # Default background
        
        if eval_data['error_line'] == line_num:
            background_color = "#ffcccc"  # Red background for error line
            
        # Escape HTML characters and add syntax highlighting
        escaped_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        # Add line
        code_html += f'<div style="background-color: {background_color}; padding: 2px 5px; white-space: pre;">'
        code_html += f'<span style="color: #888; margin-right: 10px;">{line_num}</span>{escaped_line}</div>'
    
    code_html += '</div>'
    
    # Add execution output
    output_html = f"""
    <div style="margin-top: 20px;">
        <h4>Execution Output</h4>
        <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px;">{eval_data['execution_output']}</pre>
    </div>
    """
    
    # Add explanation
    explanation = """
    <div style="margin-top: 20px;">
        <h4>Syntactic Correctness Evaluation Proces::</h4>
        <p>Executed generated model line by line. If any syntax errors are found, the execution fails. For a model to be syntactically correct, 
        it must execute without errors.</p>
    </div>
    """
    
    return HTML(header + code_html + output_html + explanation)

def visualize_constraints_comparison(demo_results):
    """Visualize constraint comparison between ground truth and generated model."""
    if 'constraint_evaluation' not in demo_results['evaluation_details']:
        return HTML("<p>Constraint evaluation data not available</p>")
    
    eval_data = demo_results['evaluation_details']['constraint_evaluation']
    
    if not eval_data['success']:
        return HTML(f"<p>Error in constraint evaluation: {eval_data['error_message']}</p>")
    
    # Get constraints from ground truth and generated model
    gt_constraints = extract_constraints(demo_results['problem']['ground_truth_model'])
    gen_constraints = extract_constraints(demo_results['pipeline_steps'][-1]['clean_code'])
    
    # Create header with overall result
    constraint_accuracy = eval_data.get('constraint_accuracy', 0)
    if not isinstance(constraint_accuracy, (int, float)):
        constraint_accuracy = 0
        
    header = f"""
    <div style="margin-bottom: 20px;">
        <h3>Constraint Evaluation</h3>
        <p><strong>Result:</strong> {constraint_accuracy:.2%} accuracy</p>
        <p>Wrong constraints: {eval_data.get('wrong_constraints', 'N/A')}</p>
        <p>Total constraints: {eval_data.get('total_constraints', 'N/A')}</p>
    </div>
    """
    
    # Create comparison table
    comparison_data = []
    constraint_comparisons = eval_data.get('constraint_comparison', [])
    
    for i, comp in enumerate(constraint_comparisons):
        gt_const = comp.get('ground_truth_constraint', '')
        gen_const = comp.get('generated_constraint', '')
        
        semantically_equivalent = comp.get('semantically_equivalent', False)
        syntactically_similar = comp.get('syntactically_similar', False)
        similarity_score = comp.get('similarity_score', 0)
        
        status = "✓" if semantically_equivalent else "✗"
        
        comparison_data.append({
            "Index": i + 1,
            "Ground Truth": gt_const if gt_const else "",
            "Generated": gen_const if gen_const else "",
            "Match": status,
            # "Similarity": f"{similarity_score:.2f}"
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Style the dataframe with colors
    def style_match(val):
        return 'background-color: lightgreen' if val == "✓" else 'background-color: lightsalmon'
    
    def style_similarity(val):
        try:
            sim = float(val)
            if sim > 0.8:
                return 'background-color: lightgreen'
            elif sim > 0.5:
                return 'background-color: khaki'
            else:
                return 'background-color: lightsalmon'
        except:
            return ''
    
    # styled_df = df.style.map(style_match, subset=['Match']).map(style_similarity, subset=['Similarity'])
    styled_df = df.style.map(style_match, subset=['Match'])
    
    # Add explanation of constraint evaluation
    explanation = """
    <div style="margin-top: 20px;">
        <h4>Constraint Evaluation Process:</h4>
        <p>Compare ach constraint in the generated model with all constraints in the ground truth model whether they represent the same restriction on the solution space</p>
        <p>To do so check if one constraint entails the other and vice versa.</p>
    </div>
    """
    
    return HTML(header + styled_df.to_html() + explanation)

def visualize_solution_comparison(demo_results):
    """Visualize solution comparison between ground truth and generated model, highlighting the validation approach."""
    if 'solution_evaluation' not in demo_results['evaluation_details']:
        return HTML("<p>Solution evaluation data not available</p>")
    
    eval_data = demo_results['evaluation_details']['solution_evaluation']
    
    if not eval_data['success']:
        return HTML(f"<p>Error in solution evaluation: {eval_data['error_message']}</p>")
    
    # Create header with overall result
    direct_match = eval_data['direct_match']
    validation_correct = eval_data.get('validation_correct', False)
    final_correct = direct_match or validation_correct
    
    header = f"""
    <div style="margin-bottom: 20px;">
        <h3>Solution Evaluation</h3>
        <p><strong>Result:</strong> {"✓ Correct" if final_correct else "✗ Incorrect"}</p>
        <p>Method 1 - Direct solution match: {"✓ Pass" if direct_match else "✗ Fail"}</p>
        <p>Method 2 - Solution validation: {"✓ Pass" if validation_correct else "✗ Fail"}</p>
    </div>
    """
    
    # Visualization of the two evaluation methods
    methods_html = """
    <div style="margin-bottom: 20px;">
        <h4>Two Complementary Evaluation Methods</h4>
        <div style="display: flex; flex-direction: row; gap: 20px;">
            <div style="flex: 1; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background-color: #f8f8f8;">
                <h5>Method 1: Direct Solution Comparison</h5>
                <p>Execute both models and directly compare their solutions</p>
                <div style="text-align: center; margin: 10px 0;">
                    <div style="display: inline-block; padding: 8px; background-color: #e1efff; border-radius: 5px; margin-bottom: 10px;">Generated Model</div>
                    <div style="margin: 5px 0;">↓ execute</div>
                    <div style="display: inline-block; padding: 8px; background-color: #e1efff; border-radius: 5px;">Solution A</div>
                </div>
                <div style="text-align: center; margin: 10px 0;">
                    <div style="display: inline-block; padding: 8px; background-color: #e1efff; border-radius: 5px; margin-bottom: 10px;">Ground Truth Model</div>
                    <div style="margin: 5px 0;">↓ execute</div>
                    <div style="display: inline-block; padding: 8px; background-color: #e1efff; border-radius: 5px;">Solution B</div>
                </div>
                <div style="text-align: center; margin: 10px 0;">
                    <div style="margin: 5px 0;">↓ compare</div>
                    <div style="display: inline-block; padding: 8px; background-color: """ + ("#d4edda" if direct_match else "#f8d7da") + """; border-radius: 5px;">""" + ("✓ Match" if direct_match else "✗ No Match") + """</div>
                </div>
            </div>
            <div style="flex: 1; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background-color: #f8f8f8;">
                <h5>Method 2: Solution Validation</h5>
                <p>Add the generated solution as a constraint to the ground truth model</p>
                <div style="text-align: center; margin: 10px 0;">
                    <div style="display: inline-block; padding: 8px; background-color: #e1efff; border-radius: 5px; margin-bottom: 10px;">Generated Model</div>
                    <div style="margin: 5px 0;">↓ execute</div>
                    <div style="display: inline-block; padding: 8px; background-color: #e1efff; border-radius: 5px;">Solution A</div>
                </div>
                <div style="text-align: center; margin: 10px 0;">
                    <div style="margin: 5px 0;">↓ add as constraint</div>
                    <div style="display: inline-block; padding: 8px; background-color: #e1efff; border-radius: 5px; margin-bottom: 10px;">Ground Truth Model + Solution A</div>
                    <div style="margin: 5px 0;">↓ solve</div>
                    <div style="display: inline-block; padding: 8px; background-color: """ + ("#d4edda" if validation_correct else "#f8d7da") + """; border-radius: 5px;">""" + ("✓ Satisfiable" if validation_correct else "✗ Unsatisfiable") + """</div>
                </div>
            </div>
        </div>
    </div>
    """
    
    # Detail the validation process that actually happens in aplai_pipeline.py
    # validation_process_html = """
    # <div style="margin-top: 20px;">
    #     <h4>Solution Validation Process in Detail</h4>
    #     <ol>
    #         <li><strong>Extract the solution</strong> from the generated model by executing it</li>
    #         <li><strong>Format the solution</strong> as variable assignments</li>
    #         <li><strong>Add these assignments</strong> as additional constraints to the ground truth model</li>
    #         <li><strong>Run the modified ground truth model</strong> to check if it remains satisfiable</li>
    #         <li>If satisfiable, the solution is <strong>valid</strong> - it satisfies all constraints in the ground truth model</li>
    #     </ol>
    # </div>
    # """
    validation_process_html = ""
    
    # Show the actual code used for validation
    validation_code_html = ""
    if 'validation_output' in eval_data and eval_data.get('generated_solution'):
        # Format the solution assignment code as it would appear in aplai_pipeline
        solution_str = str(eval_data.get('generated_solution', '[]'))
        
        validation_code_html = f"""
        <div style="margin-top: 20px;">
            <h4>Validation Code Generated</h4>
            <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto;">
# Original ground truth model
{demo_results['problem']['ground_truth_model']}

# Add solution from generated model as constraints
{make_solution_constraints(eval_data.get('generated_solution'))}

# Solve and check if satisfiable
print(f"Model Solved: {{m.solve(time_limit=10)}}, Status: {{m.status()}}")
            </pre>
            
            <h4>Validation Result</h4>
            <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px;">{eval_data.get('validation_output', 'No validation output available')}</pre>
        </div>
        """
    
    # Create solution group comparison table (from before)
    comparison_data = []
    group_comparisons = eval_data.get('group_by_group_comparison', [])
    
    for comp in group_comparisons:
        group_num = comp.get('group_number', 0)
        gt_group = comp.get('ground_truth_group', [])
        gen_group = comp.get('generated_group', [])
        
        match = comp.get('match', False)
        status = "✓" if match else "✗"
        
        comparison_data.append({
            "Group": group_num,
            "Ground Truth Solution": ", ".join(map(str, gt_group)) if gt_group else "N/A",
            "Generated Solution": ", ".join(map(str, gen_group)) if gen_group else "N/A",
            "Match": status
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Style the dataframe with colors
    def style_match(val):
        return 'background-color: lightgreen' if val == "✓" else 'background-color: lightsalmon'
    
    styled_df = df.style.map(style_match, subset=['Match'])
    
    # Add explanation of solution evaluation, including paper formula
    # explanation = """
    # <div style="margin-top: 20px; padding: 15px; background-color: #f9f9f9; border-radius: 5px;">
    #     <h4>How This Aligns with the Paper</h4>
    #     <p>The solution validation approach directly implements the paper's formula:</p>
    #     <div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; font-style: italic; margin: 10px 0;">
    #         acc<sub>sol</sub> = (1/N) Σ valid(a<sub>i</sub>, true<sub>i</sub>)
    #     </div>
    #     <p>Where valid(a<sub>i</sub>, true<sub>i</sub>) equals 1 if the solution a<sub>i</sub> from the predicted model satisfies all constraints in the ground truth model.</p>
    #     <p>This is exactly what Method 2 checks: we take the solution from the generated model and verify if it satisfies the ground truth model by adding it as a constraint and checking satisfiability.</p>
    #     <p>For optimization problems, this would additionally check if the objective value is optimal.</p>
    # </div>
    # """
    explanation = """
    <div style="margin-top: 20px; padding: 15px; background-color: #f9f9f9; border-radius: 5px;">
        <p>Solution accuracy is computed as the percentage of problems for which the predicted solution is correct.</p>
        <p>The solution is considered correct if it is equivalent to the ground truth solution OR if it is otherwise a correct solution (check by adding the solution as constraints to the ground truth problem).</p>
    </div>
    """
    
    # Combining all parts
    if len(comparison_data) > 0:
        solution_table = "<h4>Solution Comparison</h4>" + styled_df.to_html()
    else:
        solution_table = ""
        
    return HTML(header + methods_html + validation_process_html + validation_code_html + solution_table + explanation)

def make_solution_constraints(solution):
    """Format solution as constraint assignments as done in aplai_pipeline.py."""
    if solution is None:
        return "# No solution available to add as constraints"
    
    constraints = []
    try:
        # Handle different solution formats
        if isinstance(solution, list):
            # For list of groups format
            for i, group in enumerate(solution):
                if isinstance(group, list):
                    for value in group:
                        constraints.append(f"m += group_{i} == {value}")
        else:
            # For dictionary format (common in aplai_pipeline)
            var_names = get_var_names_from_exec_model_str("m = Model()\n" + "\n".join([f"{k} = intvar(0, 10)" for k in solution.keys()]))
            for var in var_names:
                if var in solution:
                    constraints.append(f"m += {var} == {solution[var]}")
    except:
        # Fallback when format is unexpected
        constraints.append(f"# Could not format solution constraints: {solution}")
    
    return "\n".join(constraints) if constraints else "# No constraints generated from solution"

def visualize_model_equivalence(demo_results):
    """Visualize model equivalence evaluation."""
    if 'model_equivalence' not in demo_results['evaluation_details']:
        return HTML("<p>Model equivalence evaluation data not available</p>")
    
    eval_data = demo_results['evaluation_details']['model_equivalence']
    
    if not eval_data['success']:
        return HTML(f"<p>Error in model equivalence evaluation: {eval_data['error_message']}</p>")
    
    # Create header with overall result
    equivalent = eval_data['equivalent']
    
    header = f"""
    <div style="margin-bottom: 20px;">
        <h3>Model Equivalence Evaluation</h3>
        <p><strong>Result:</strong> {"✓ Equivalent" if equivalent else "✗ Not Equivalent"}</p>
    </div>
    """
    
    # Show the models side by side
    gt_model = demo_results['problem']['ground_truth_model']
    gen_model = demo_results['pipeline_steps'][-1]['clean_code']
    
    models_html = f"""
    <div style="display: flex; margin-top: 20px;">
        <div style="flex: 1; margin-right: 10px;">
            <h4>Ground Truth Model</h4>
            <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; max-height: 400px; overflow: auto;">{gt_model}</pre>
        </div>
        <div style="flex: 1; margin-left: 10px;">
            <h4>Generated Model</h4>
            <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; max-height: 400px; overflow: auto;">{gen_model}</pre>
        </div>
    </div>
    """
    
    # Add explanation of model equivalence
    explanation = """
    <div style="margin-top: 20px;">
        <p>Check if the generated model and ground truth model represent the same constraint satisfaction problem.</p>
        <ol>
            <li>have the same solution space</li>
            <li>constraints in one are entailed by the other, and vice versa</li>
        </ol>
        <p>Create a combined model that is satisfiable only if the two models have different solutions (essentially use XOR). 
        If this combined model is unsatisfiable, the original models are equivalent.</p>
    </div>
    """
    
    # The key idea is to check whether there exists any assignment that, under the candidate variable mapping, makes the two sets of constraints behave differently. This is achieved by creating a final model with the constraint:
    
    return HTML(header + models_html + explanation)

def visualize_overall_metrics(demo_results):
    """Visualize overall metrics as a summary dashboard."""
    metrics = demo_results['evaluation_metrics']
    
    # Extract metric values
    syntactic_correct = metrics.get('syntactic_correctness', {}).get('correct', False)
    solution_correct = metrics.get('solution_evaluation', {}).get('correct', False)
    validation_correct = metrics.get('solution_validation', {}).get('correct', False)
    model_equivalent = metrics.get('model_equivalence', {}).get('equivalent', False)
    
    # Safely get constraint accuracy
    constraint_accuracy = 0
    if "constraint_evaluation" in metrics:
        constraint_accuracy = metrics["constraint_evaluation"].get("constraint_accuracy", 0)
        if not isinstance(constraint_accuracy, (int, float)):
            constraint_accuracy = 0
    
    # Format for display
    final_solution_correct = solution_correct or validation_correct
    
    # Create metrics table
    metrics_data = [
        {"Metric": "Syntactic Correctness", "Result": "✓" if syntactic_correct else "✗", "Description": "Code runs without errors"},
        {"Metric": "Solution Correctness", "Result": "✓" if solution_correct else "✗", "Description": "Direct match with ground truth solution"},
        {"Metric": "Solution Validation", "Result": "✓" if validation_correct else "✗", "Description": "Generated solution is valid for ground truth model"},
        {"Metric": "Final Solution Correctness", "Result": "✓" if final_solution_correct else "✗", "Description": "Either direct match or validation passes"},
        {"Metric": "Model Equivalence", "Result": "✓" if model_equivalent else "✗", "Description": "Models have identical solution spaces"},
        {"Metric": "Constraint Accuracy", "Result": f"{constraint_accuracy:.2%}", "Description": "Proportion of correctly modeled constraints"}
    ]
    
    df = pd.DataFrame(metrics_data)
    
    # Style the dataframe
    def style_result(val):
        if val == "✓":
            return 'background-color: lightgreen'
        elif val == "✗":
            return 'background-color: lightsalmon'
        elif isinstance(val, str) and '%' in val:
            # For percentage values
            try:
                pct = float(val.replace('%', '')) / 100
                if pct >= 0.8:
                    return 'background-color: lightgreen'
                elif pct >= 0.5:
                    return 'background-color: khaki'
                else:
                    return 'background-color: lightsalmon'
            except:
                return ''
        return ''
    
    styled_df = df.style.map(style_result, subset=['Result']) 
    
    # Create chart data
    labels = ['Syntactic', 'Solution', 'Model', 'Constraint']
    values = [
        1.0 if syntactic_correct else 0.0,
        1.0 if final_solution_correct else 0.0,
        1.0 if model_equivalent else 0.0,
        float(constraint_accuracy)
    ]
    
    # Plot chart
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=['#5cb85c' if v >= 0.5 else '#d9534f' for v in values])
    ax.set_ylim(0, 1)
    ax.set_title('Evaluation Metrics Summary')
    ax.set_ylabel('Score (1 = Pass, 0 = Fail)')
    
    # Add text labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                'Pass' if height >= 0.5 else 'Fail',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Convert plot to HTML
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    chart_html = f'<img src="data:image/png;base64,{img_str}" alt="Metrics Chart">'
    
    # Add explanation
    explanation = """
    <div style="margin-top: 20px;">
        <h4>Evaluation Metrics Explained</h4>
        <p><strong>Syntactic Correctness</strong>: Checks if the generated code can be executed without syntax errors.</p>
        <p><strong>Solution Correctness</strong>: Checks if the generated model produces the same solution as the ground truth model.</p>
        <p><strong>Solution Validation</strong>: Verifies if the solution from the generated model satisfies the ground truth model.</p>
        <p><strong>Model Equivalence</strong>: Determines if both models represent the same constraint problem with identical solution spaces.</p>
        <p><strong>Constraint Accuracy</strong>: Measures what proportion of the constraints are correctly modeled.</p>
    </div>
    """
    
    # Combine table and chart
    return HTML(styled_df.to_html() + f'<div style="margin-top: 20px;">{chart_html}</div>' + explanation)

def visualize_constraint_matching_process(demo_results):
    """Visualize the detailed constraint matching process with step-by-step explanation."""
    if 'constraint_evaluation' not in demo_results['evaluation_details']:
        return HTML("<p>Constraint evaluation data not available</p>")
    
    eval_data = demo_results['evaluation_details']['constraint_evaluation']
    
    if not eval_data.get('success', False):
        return HTML(f"<p>Error in constraint evaluation: {eval_data.get('error_message', 'Unknown error')}</p>")
    
    try:
        # Get constraints from both models
        gt_constraints = extract_constraints(demo_results['problem']['ground_truth_model'])
        gen_constraints = extract_constraints(demo_results['pipeline_steps'][-1]['clean_code'])
        
        # Create step-by-step explanation
        # steps_html = """
        # <div style="margin-top: 20px;">
        #     <h4>Constraint Matching Process - Step by Step</h4>
        #     <ol>
        #         <li><strong>Extract Constraints</strong>: First, we extract all constraint statements from both models.</li>
        #         <li><strong>Compare Each Constraint</strong>: For each constraint in the ground truth model, we try to find an equivalent constraint in the generated model.</li>
        #         <li><strong>Determine Equivalence</strong>: Two constraints are equivalent if they represent the same restriction on the solution space.</li>
        #         <li><strong>Calculate Accuracy</strong>: The constraint accuracy is the percentage of ground truth constraints that have an equivalent match in the generated model.</li>
        #     </ol>
        # </div>
        # """
        steps_html = ""
        
        # Create a visual representation of the matching process
        matching_html = """
        <div style="margin-top: 20px;">
            <h4>Constraint Matching Visualization</h4>
            <div style="display: flex; align-items: flex-start;">
        """
        
        # Left side: Ground truth constraints
        matching_html += """
            <div style="flex: 1; margin-right: 10px;">
                <h5>Ground Truth Constraints</h5>
                <ul style="list-style-type: none; padding-left: 0;">
        """
        
        for i, constraint in enumerate(gt_constraints):
            matching_html += f'<li style="margin-bottom: 10px; padding: 5px; background-color: #f0f8ff; border-radius: 5px;">{i+1}. {constraint}</li>'
        
        matching_html += """
                </ul>
            </div>
        """
        
        # Middle: Arrows showing matches
        matching_html += """
            <div style="width: 50px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
        """
        
        # Calculate matches - this is an approximation based on the available information
        wrong_constraints = eval_data.get('wrong_constraints', 0)
        correct_constraints = len(gt_constraints) - wrong_constraints
        
        for i in range(len(gt_constraints)):
            if i < correct_constraints:
                matching_html += '<div style="margin: 17px 0; color: green;">→</div>'
            else:
                matching_html += '<div style="margin: 17px 0; color: red;">✗</div>'
        
        matching_html += """
            </div>
        """
        
        # Right side: Generated constraints
        matching_html += """
            <div style="flex: 1; margin-left: 10px;">
                <h5>Generated Constraints</h5>
                <ul style="list-style-type: none; padding-left: 0;">
        """
        
        for i, constraint in enumerate(gen_constraints):
            # Color based on match estimation
            if i < correct_constraints:
                bg_color = "#e6ffe6"  # Light green for matched
            else:
                bg_color = "#ffebeb"  # Light red for unmatched
                
            matching_html += f'<li style="margin-bottom: 10px; padding: 5px; background-color: {bg_color}; border-radius: 5px;">{i+1}. {constraint}</li>'
        
        matching_html += """
                </ul>
            </div>
        """
        
        matching_html += """
            </div>
        </div>
        """
        
        # Add explanation of the actual algorithm used
        algorithm_explanation = f"""
        <div style="margin-top: 20px; padding: 15px; background-color: #f9f9f9; border-radius: 5px;">
            <p>Uses CPMPy to check semantic equivalence:</p>
            <ol>
                <li>Converted each constraint to a CPMPy representation</li>
                <li>For each GT constraint, look for a generated/predicted constraint that is logically equivalent</li>
                <li>Logical equivalence - one constriant entails the other, and vice versa</li>
                <li>Result: <strong>{wrong_constraints}</strong> constraints out of <strong>{len(gt_constraints)}</strong> are wrong</li>
                <li>Constraint Accuracy: <strong>{(len(gt_constraints) - wrong_constraints) / len(gt_constraints) if len(gt_constraints) > 0 else 0:.2%}</strong></li>
            </ol>
            <p>Accuracy is computed by 1 - the percentage of wrong constraints. Where wrong constraints are either false positives (generated constraints that were predicted wrong - do not have an equivalent counterpart in the ground truth model) or false negatives (constraints that were missed - GT constraints that do not have an equivalent counterpart in the generated model). The number of wrong constraints is bounded by the total number of constraints.</p>
        </div>
        """
        
        return HTML(steps_html + matching_html + algorithm_explanation)
    except Exception as e:
        return HTML(f"<p>Error in constraint matching visualization: {str(e)}</p>")