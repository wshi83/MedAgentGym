import os
from .base import AbstractEHRTask
import json

overall_information = """You are a scientist conducting biomedical research and constantly facing statistical problems. Sometimes, you need to find the minimum sample size to achieve a specific power. In other times, you would like to know the statistical power given a population size.
"""

class NPowerAITask(AbstractEHRTask):
    """
    Generic task for answering questions based on the NPower.

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
        return f"EHRGym.npower-ai.{formatted_name}"
    
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
            if self.mode == "test":
                task_file = "test_tasks.jsonl"
            else: 
                task_file = 'train_tasks.jsonl'
            task_path = os.path.join(self.data_path, task_file)
            self.task_list = []
            with open(task_path, 'r') as f:
                for line in f:
                    self.task_list.append(json.loads(line))
        task_data = self.task_list[self.task_id]
        self.context = task_data['task_type']
        self.target_type = task_data['estimate_target'] # power or size
        self.question = task_data['question']
        self.answer = task_data['answer']

        # configure the task
        goal, info = self.setup_goal()
        return goal, info

    
    def setup_goal(self) -> tuple[str, dict]:
        """
        Set up the goal for the task
        """
        super().setup_goal()
        # get the task configuration
        self.goal = f"Write a python code to solve the a statistic question. Use the variable 'answer' to store the answer of the code.\nQuestion: {self.question}\n"
        info = {}
        return self.goal, info

    def _get_obs(self) -> dict:
        obs = {}
        obs["type"] = "initial_observation"
        obs["info"] = {}
        obs["info"]["overall"] = overall_information
        obs["info"]["task_goal"] = self.goal
        obs["info"]["instruction"] = overall_information
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
            pred = obs["env_message"].strip()
            if type(self.answer) == list:
                ans = self.answer[0]
            else:
                ans = self.answer
            # ans = self.answer

            correctness = False

            if self.target_type == 'size':
                try:
                    if not isinstance(pred, int): 
                        pred = int(pred)

                except Exception as e:
                            error_msg = f"The answer of size should be able to convert to an integer: {e}"
                            print(error_msg)
                            return (
                                0,
                                False,
                                error_msg,
                                {"message": error_msg}
                            )
                if pred == self.answer:
                    correctness = True

            else: # is a power question
                try:
                    if not isinstance(pred, float): 
                        pred = float(pred)

                except Exception as e:
                            error_msg = f"The answer of power should be able to convert to a float: {e}"
                            print(error_msg)
                            return (
                                0,
                                False,
                                error_msg,
                                {"message": error_msg}
                            )
                if pred > self.answer * 0.99 and pred < self.answer * 1.01:  # plus minus 1% of wiggle room
                    correctness = True


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
    