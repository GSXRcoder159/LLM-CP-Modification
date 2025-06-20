from typing import Dict
from dataclasses import dataclass
import numpy as np


class APLAIProblem:
    def __init__(self, question, description, cpmpy_code, pseudo_model=None, direct_solution=None, entities_as_str=None):
        self.question = question
        self.description = description
        self.cpmpy_code = cpmpy_code
        self.cpmpy_code_wrapped = "```python\n" + cpmpy_code + "\n```"
        self.pseudo_model = pseudo_model if pseudo_model else "Not available"
        self.direct_solution = direct_solution if direct_solution else "Not available"
        self.entities_as_str = entities_as_str if entities_as_str else "Not available"


class Nl4optProblem:
    def __init__(self, document, order_mapping, cpmpy_code, solution, canonical, pseudo_model, direct_solution, entities=None):
        self.question = document
        self.order_mapping = order_mapping
        self.cpmpy_code = cpmpy_code
        self.cpmpy_code_wrapped = "```python\n" + cpmpy_code + "\n```"
        self.solution = solution
        self.canonical = canonical
        self.pseudo_model = pseudo_model
        self.direct_solution = direct_solution
        self.entities = entities
        self.entities_as_str = ''
        if entities:
            for entity in entities:
                self.entities_as_str += entity['entity_group'] + ' (' + str(entity['start']) + '-' + str(
                    entity['end']) + '): ' + entity['word'] + '\n'


class Puzzle:
    def __init__(self, question_: str, cpmpy_model_: str, pseudo_model_: str, answer_: [str],
                 entities_as_str_: str = '', direct_solution_: str = ''):
        self.question = question_
        self.pseudo_model = pseudo_model_
        self.cpmpy_model = cpmpy_model_
        self.cpmpy_code_wrapped = "```python\n" + cpmpy_model_ + "\n```"
        self.answer_actual = answer_
        self.entities_as_str = entities_as_str_
        self.direct_solution = direct_solution_

    def format_question(self):
        return self.question


@dataclass
class CanonicalFormulation:
    def __init__(self, objective, constraints):
        self.objective = objective
        self.constraints = constraints


class CpmpyModelWithVars:
    """ Wrapper for combining a CPMpy model with the variables."""
    def __init__(self, model, vars_list):
        self.model = model
        self.vars_ = vars_list


class ModificationExample:
    """Class to represent a constraint programming model modification example."""
    
    def __init__(self, key, question, cpmpy_code, modification_request, modified_cpmpy_code, 
                 pseudo_model=None, entities_as_str=None):
        """
        Initialize a model modification example.
        
        Args:
            question: The original problem description
            cpmpy_code: The original CP model
            modification_request: The natural language description of the modification
            modified_cpmpy_code: The ground truth modified CP model
            pseudo_model: Optional pseudo model
            entities_as_str: Optional named entities
        """
        self.key = key
        self.question = question
        self.cpmpy_code = cpmpy_code
        self.cpmpy_code_wrapped = "```python\n" + cpmpy_code + "\n```"
        self.modification_request = modification_request
        self.modified_cpmpy_code = modified_cpmpy_code
        self.modified_cpmpy_code_wrapped = "```python\n" + modified_cpmpy_code + "\n```"
        self.pseudo_model = pseudo_model if pseudo_model else "Not available"
        self.entities_as_str = entities_as_str if entities_as_str else "Not available"
        self.direct_solution = "Not available"

        
    def format_for_example(self):
        """Format the example for use in prompts."""
        return {
            "question": self.question,
            "existing_model": self.cpmpy_code_wrapped,
            "modification_request": self.modification_request,
            "modified_cpmpy_code": self.modified_cpmpy_code_wrapped
        }
    
    def __eq__(self, value):
        if isinstance(value, ModificationExample):
            return self.key == value.key
        return False
    def __hash__(self):
        return hash(self.key)
