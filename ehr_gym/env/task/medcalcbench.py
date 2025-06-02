import os
from .base import AbstractEHRTask
import json

overall_information = """
You work in a hospital, and a common task in your work is to calculate some biological values of your patients. To do this, you need to identify from clinical notes what information is relevant, before using your clinical knowledge to calculate.
Instructions to calculate is listed below {}.
"""

instruction = """You work in a hospital, and a common task in your work is to calculate some biological values of your patients. 
To do this, you need to identify from clinical notes what information is relevant, before using your clinical knowledge to calculate.
And then write a Python code to calculate the value.
In the code, please use the variable 'answer' to store the answer of the code.
In the main function, please print the final answer of the code without any other text.
"""
# Hints to calculate is listed below: 
# {overall}.
# """

class MedCalBenchTask(AbstractEHRTask):
    """
    Generic task for answering questions based on the MedCalcBench EHR data.

    Class sttributed:
    -----------------
    config_path: str
        Path to the configuration file
    
    Parameters:
    -----------------
    task_id: int
        The id of the task inside the data.
    
    """
    permitted_actions = ['validate_code', 'debug', 'terminal']
    def __init__(
        self,
        task_id: int,
        data_path: str = None,
        calculator_instruction_path: str = None,
        debugger_config: dict = None,
        mode: str = "test",
    ) -> None:
        super().__init__(task_id=task_id)
        self.task_id = task_id
        self.task_list = None
        self.data_path = data_path
        self.calculator_instruction_path = calculator_instruction_path
        self.debugger_config = debugger_config
        self.mode = mode
    
    @classmethod
    def get_task_id(cls):
        # Get the class name and remove the word 'Task' from the end if it exists
        class_name = cls.__name__.replace("Task", "")
        # Convert CamelCase to hyphen-separated format
        formatted_name = "".join(
            ["-" + c.lower() if c.isupper() else c for c in class_name]
        ).lstrip("-")
        return f"EHRGym.medcalcbench.{formatted_name}"
    
    def setup(self) -> tuple[str, dict]:
        """
        Set up the task

        Parameters:
        -----------------
        data_path: str
            Path to the data directory
        """

        # locate the task
        if self.task_list is None:
            if self.mode == 'test':
                task_file = 'test_tasks.jsonl'
            else:
                task_file = 'train_tasks_all.jsonl'
            task_path = os.path.join(self.data_path, task_file)
            self.task_list = []
            with open(task_path, 'r') as f:
                for line in f:
                    self.task_list.append(json.loads(line))
        task_data = self.task_list[self.task_id]
        self.context = task_data['Patient Note']
        self.question = task_data['Question']
        self.answer = task_data['Ground Truth Answer']
        self.calculator = task_data['Calculator Name']
        calculator_path = os.path.join(self.data_path, "calculation_method.jsonl")
        self.calculator_info = {}
        with open(calculator_path, 'r') as f:
            for line in f:
                calc_data = json.loads(line)
                if not calc_data['Calculator'] in self.calculator_info:
                    self.calculator_info[calc_data['Calculator']] = calc_data["Short Summary"]
        self.calculator_instruction = self.calculator_info[self.calculator]

        # configure the task
        goal, info = self.setup_goal()
        return goal, info

    
    def setup_goal(self) -> tuple[str, dict]:
        """
        Set up the goal for the task
        """
        super().setup_goal()
        # get the task configuration
        self.goal = f"Write a python code to solve the given question. Use the variable 'answer' to store the answer of the code.\nQuestion: {self.question}\n"
        info = {}
        return self.goal, info

    def _get_obs(self) -> dict:
        obs = {}
        obs["type"] = "initial_observation"
        obs["info"] = {}
        obs["info"]["overall"] = overall_information.format(self.calculator_instruction)
        obs["info"]["task_goal"] = self.goal
        obs["info"]["instruction"] = instruction # .format(overall=self.calculator_instruction)
        return obs
    

    def validate(self, chat_messages, obs):
        """
        Validate the task

        Parameters:
        -----------------
        chat_messages: list
            List of chat messages
        obs: dict
            Observation dictionary
        """
        
        if obs["type"] == "code_execution":
            pred = obs["env_message"]
            if type(self.answer) == list:
                ans = self.answer[0]
            else:
                ans = self.answer
            # ans = self.answer

            correctness = False
            try:
                if float(self.answer) >= float(pred) * 0.95 or float(self.answer) <= float(pred) * 1.05:
                    # plus minus 5% as the original tolerance
                    correctness = True
            except Exception as e:
                return (
                    0,
                    False,
                    "The code encountered with errors",
                    {"message": f"The code encountered with errors during evaluation. There seems to be something wrong with the final answer or not print it. Can you check the error message and try to fix it?\nError Message: {str(e)}"}
                )


            if correctness:
                return (
                    1, 
                    True, 
                    "The answer is correct", 
                    {"message": "The question is correctly solved."}
                )
            else:
                return (
                    0,
                    False,
                    "The answer is incorrect",
                    {"message": "The question is not correctly solved. Can you think about whether there might be some mistakes in the previous code?"}
                )
        elif obs["type"] == "error_message":
            return (
                0,
                False,
                "The code encountered with errors",
                {"message": f"The code encountered with errors. Can you check the error message and try to fix it?\nError Message: {obs['message']}"}
            )
        else:
            return (
                0,
                False,
                "",
                {"message": obs['env_message']}
            )
    