import os
from datetime import datetime
import numpy as np
import json

from llms4cp.call_llms import call_llm
from llms4cp.data_reader import get_puzzles
from llms4cp.dataset_classes import Puzzle
from llms4cp.lgps_cpmpy_model_equivalence import check_cpmpy_str_models_equivalence, get_cpmpy_str_per_const_eval, is_objective_equivalent
from llms4cp.util import run_code, get_examples_for_context_modification, choose_example_selector_modification, execute_and_get_solution, \
    cpmpy_model_is_unsat, get_number_of_constraints, get_var_names_from_exec_model_str, count_unchanged_constraints
from llms4cp.in_context_config import *

def pipeline_modify_model(dataset, all_examples, dataset_name, with_pseudo=False, with_ner=False):
    """
    Pipeline for modifying existing constraint models based on natural language modification requests.
    
    Args:
        dataset: List of examples, where each example contains original problem, existing model, 
                modification request, and ground truth modified model
        all_examples: List of all context examples
        dataset_name: Name of the dataset (e.g., APLAI, NL4OPT, LGPs)
        with_pseudo: Whether to include pseudo model modification step
        with_ner: Whether to include named entity recognition
    """
    const_wrong, const_total, const_err = 0, 0, 0
    prob_wrong, prob_err = 0, 0
    sol_wrong, sol_err = 0, 0
    unchanged_constraints_total = 0

    to_log = []

    method = 'CPMPY'
    if with_ner:
        method = 'PSEUDO'
    elif with_pseudo:
        method = 'PSEUDO'

    # TODO: add support for loading examples
    example_selector = choose_example_selector_modification(all_examples)

    for i, example in enumerate(dataset):
        print(f"Problem {i + 1} / {len(dataset)}")

        if i >= LIMIT:
            break
        
        # get the index of the current example in the all_examples list
        example_index = all_examples.index(example)
        print(f"Example index in all_examples: {example_index}")
        
        if example_index == -1:
            print(f"Example not found in all_examples: {example}")

        # Extracting the required fields from the example
        question = example.question
        existing_model = example.cpmpy_code
        modification_request = example.modification_request
        ground_truth_modified_model = example.modified_cpmpy_code
        
        # Prepare input variables
        input_vars = {
            "question": question,
            "existing_model": existing_model,
            "modification_request": modification_request
        }
        
        # Construct the full input
        full_input = f"{question}\n\n### ORIGINAL MODEL:\n{existing_model}\n\n### CHANGE REQUEST:\n{modification_request}"
        
        pseudo_model_answer = 'N/A'
        
        if with_ner and hasattr(example, 'entities_as_str'):
            input_vars["ner"] = example.entities_as_str
            full_input += '\n' + example.entities_as_str
        try:
            # Optionally get a pseudo model modification first
            # if with_pseudo:
            #     messages, pseudo_model_answer = call_llm(
            #         get_examples_for_context(example_selector, input_vars, dataset),
            #         full_input, system_message=SYSTEM_MESSAGE_MODIFY_PSEUDO, method=method, step='GEN_PSEUDO')
            #     print(f'ANSWER (PSEUDO MODEL MODIFICATION):\n{pseudo_model_answer}')
            #     input_vars["pseudo_model"] = pseudo_model_answer
            #     full_input += '\n\n### PSEUDO MODEL MODIFICATION:\n' + pseudo_model_answer

            # print(get_examples_for_context_modification(example_selector, input_vars, all_examples, i))
            # Get the modified CPMpy model
            messages, modified_model = call_llm(
                # get_examples_for_context_modification(example_selector, input_vars, all_examples, example_index),
                [],
                full_input, system_message=SYSTEM_MESSAGE_MODIFY_CPMPY, method=method)
            
#             modified_model = """```python
# from cpmpy import *
# import json

# # Parameters
# n = 20  # total number of steps in the stair
# m1, m2 = 3, 5  # number of steps that can be taken at a time

# # Decision variables
# steps = intvar(0, m2, shape=n) # steps taken at each move

# # Model setup
# m = Model()

# # Constraint: the sum of steps should equal the total number of stairs
# m += sum(steps) == n

# # Constraint: the number of steps taken at each move should be between m1 and m2 or 0
# m += [(steps[i] >= m1) | (steps[i] == 0) for i in range(n)]
# m += [steps[i] <= m2 for i in range(n)]

# # Trailing zeros: If a step is 0, then all the following steps should be 0
# for i in range(1, n):
#     m += (steps[i - 1] == 0).implies(all(steps[j] == 0 for j in range(i, n)))

# # Constraint: at least one 5-step move must be used
# m += sum(steps[i] == m2 for i in range(n)) >= 1

# # Solve the model and print the results in the required format
# if m.solve():
#     solution = {"steps": [steps[i].value() for i in range(n) if steps[i].value() != 0]}
#     print(json.dumps(solution))
# ```"""

            print(f'MODIFIED CPMPY MODEL:\n{modified_model}\n')

        except Exception as e:
            print("Error while calling LLM:" + str(e))
            continue

        # SOLUTION EVALUATION 1 (check if modified model produces correct solution)
        # TODO: extend and modify solution evaluation according to piepelines (not each pipeline uses a different approach)
        try:
            # remove wrappers
            if '```python' in modified_model:
                code = modified_model.split('```python')[1].split('```')[0]
            elif '```' in modified_model:
                code = modified_model.split('```')[1].split('```')[0]
            else:
                code = modified_model
            
            if '```python' in ground_truth_modified_model:
                ground_truth_code = ground_truth_modified_model.split('```python')[1].split('```')[0]
            elif '```' in ground_truth_modified_model:
                ground_truth_code = ground_truth_modified_model.split('```')[1].split('```')[0]
            else:
                ground_truth_code = ground_truth_modified_model

            actual_solution = run_code(ground_truth_code)
            print(f'ACTUAL SOLUTION:\n{actual_solution}\n')

            solution = run_code(code)
            print(f'SOLUTION:\n{solution}\n')

            # compare
            if solution.strip() == actual_solution.strip():
                sol_log = 'True'
                print("Solutions are equivalent")
            else:
                if solution is None:
                    sol_2 = cpmpy_model_is_unsat(code)
                    if sol_2:
                        print("Solution 2 successful")
                        sol_log = 'True'
                    else:
                        print("Solution 2 failed")
                        sol_wrong += 1
                        sol_log = 'False'
                else:
                    # predicted solution is a json object
                    solution = json.loads(solution)
                    # get keys
                    keys = list(solution.keys())
                    # add each key as a constraint in code

                    _consts_to_add = ''
                    for key in keys:
                        _consts_to_add += f'm += {key} == {solution[key]}\n'

                    new_code = code + '\n' + _consts_to_add + '\n'
                    new_code += 'print(f"Model Solved: {m.solve(time_limit=10)}, Status: {m.status()}")\n'
                    code_output = run_code(new_code)
                    sol_2 = "Model Solved: True" in code_output
                    if sol_2:
                        print("Solution 2 successful")
                        sol_log = 'True'
                    else:
                        print("Solution 2 failed")
                        sol_wrong += 1
                        sol_log = 'False'
        except Exception as e:
            print("Error when trying to check solution from modified model: " + str(e))
            sol_log = 'Error when trying to check solution from modified model: ' + str(e)
            sol_err += 1

        # CONSTRAINT EVALUATION
        try:
            keep_only_last_line = True if dataset_name == 'APLAI' else False
            wrong_consts, total_consts = get_cpmpy_str_per_const_eval(code, ground_truth_code, keep_only_last_line=keep_only_last_line)
            const_wrong += wrong_consts
            const_total += total_consts
            # if there is m.minimize or m.maximize in the ground truth model, also check for objectives
            if 'm.minimize' in ground_truth_code or 'm.maximize' in ground_truth_code:
                is_obj_correct = is_objective_equivalent(code, ground_truth_code)
                const_log = f'Wrong: {wrong_consts}, Total: {total_consts}, Is objective correct: {is_obj_correct}'
            else:
                const_log = f'Wrong: {wrong_consts}, Total: {total_consts}'
            print(const_log) 
        except Exception as e:
            print("Error in constraint evaluation: " + str(e))
            const_log = 'Error in constraint evaluation: ' + str(e)
            n_consts = get_number_of_constraints(ground_truth_code)
            const_err += n_consts
            const_total += n_consts

        # MODEL EVALUATION
        try:
            if check_cpmpy_str_models_equivalence(code, ground_truth_code):
                print("Models are equivalent")
                mod_log = 'True'
            else:
                print("Models are not equivalent")
                prob_wrong += 1
                mod_log = 'False'
        except Exception as e:
            print("Error in model evaluation: " + str(e))
            mod_log = 'Error in model evaluation: ' + str(e)
            prob_err += 1

        # Count unchanged constraints
        try:
            unchanged = count_unchanged_constraints(existing_model, code, ground_truth_code)
            unchanged_constraints_total += unchanged
            unchanged_log = f'Unchanged: {unchanged}'
            print(unchanged_log)
        except Exception as e:
            print("Error counting unchanged constraints: " + str(e))
            unchanged_log = 'Error counting unchanged constraints: ' + str(e)

        print("-------------------------------------------------------------")
        to_log.append((question, existing_model, modification_request, pseudo_model_answer, modified_model, 
                      ground_truth_modified_model, sol_log, mod_log, const_log, unchanged_log))

    total = len(dataset)
    sol_acc = 100.0 - ((sol_wrong + sol_err) * 100.0 / total) if total > 0 else 0
    const_acc = 100.0 - ((const_wrong + const_err) * 100.0 / const_total) if const_total > 0 else 0
    mod_acc = 100.0 - ((prob_wrong + prob_err) * 100.0 / total) if total > 0 else 0
    avg_unchanged = unchanged_constraints_total / total if total > 0 else 0

    print(f"Solution accuracy: {sol_acc}%")
    print(f'Constraint accuracy: {const_acc}%')
    print(f'Model accuracy: {mod_acc}%')
    print(f'Average unchanged constraints: {avg_unchanged}')
    print(f'Wrong solutions: {sol_wrong}, total: {total}')
    print(f'Wrong constraints: {const_wrong}, total: {const_total}')
    print(f'Wrong models: {prob_wrong}, total: {total}')
    print(f'Errors: solution-level: {sol_err}, constraint-level: {const_err}, model-level: {prob_err}')

    pref = 'pseudo_mod' if with_pseudo else 'cpmpy_mod'
    pref = 'ner_mod' if with_ner else pref

    if EXAMPLES_SELECTOR == 'mmr' or EXAMPLES_SELECTOR == 'sim':
        examples_pref = 'reversed_' if REVERSED_ORDER_ICL else 'normal_'
        examples_pref += EXAMPLES_SELECTOR
    else:
        examples_pref = EXAMPLES_SELECTOR

    path_to_results = f'results/model_modification/{pref}/{NUM_EXAMPLES}-shot/{examples_pref}/{MODEL}'
    if not os.path.exists(path_to_results):
        os.makedirs(path_to_results)
    with open(f'{path_to_results}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt', 'w', encoding="utf-8") as f:
        f.write(f"Solution accuracy: {sol_acc}%\n")
        f.write(f'Constraint accuracy: {const_acc}%\n')
        f.write(f'Model accuracy: {mod_acc}%\n')
        f.write(f'Average unchanged constraints: {avg_unchanged}\n\n')
        f.write(f'Wrong solutions: {sol_wrong}, total: {total}\n')
        f.write(f'Wrong constraints: {const_wrong}, total: {const_total}\n')
        f.write(f'Wrong models: {prob_wrong}, total: {total}\n')
        f.write(f'Errors: solution-level: {sol_err}, constraint-level: {const_err}, model-level: {prob_err}\n')
        f.write('---------------------------------------------------------------------------------------------\n\n')
        for item in to_log:
            for entry in item:
                f.write(f'{entry}\n')
            f.write('---------------------------------------------------------------------------------------------\n\n')


def run_model_modification_pipeline(dataset, all_examples, dataset_name, method='CPMPY'):
    """
    Run the model modification pipeline with the specified method
    
    Args:
        dataset: List of examples with modification requests
        all_examples: List of all context examples
        method: The method to use (CPMPY, PSEUDO, NER)
    """
    if method == 'CPMPY':
        print('Running CPMPy model modification pipeline')
        pipeline_modify_model(dataset, all_examples, dataset_name, with_pseudo=False)  # CPMpy
    elif method == 'PSEUDO':
        print('Running Pseudo model modification pipeline')
        pipeline_modify_model(dataset, all_examples, dataset_name, with_pseudo=True)  # Pseudo and Cpmpy
    elif method == 'NER':
        print('Running NER model modification pipeline')
        pipeline_modify_model(dataset, all_examples, dataset_name, with_pseudo=True, with_ner=True)  # NER and Pseudo and Cpmpy
    else:
        raise ValueError(f"Not supported method: {method}")