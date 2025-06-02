import os
from .base import AbstractEHRTask
import json
from ehr_gym.env.action.function import validate_code

context_information = """The context code around the function is as follows:
<code>
{context}
</code>
"""

signature_information = """Please write a Python function with the following signature:
{signature}
"""

instruction = """You are an biomedical expert in writing bioinformatics code and answer questions accordingly. 
Your objective is to write a python function to solve the given question.
Please only write the function, do not include any other text.
{signature_info}
"""

class BiocoderTask(AbstractEHRTask):
    """
    Generic task for answering questions based on the Biocoder data.

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
        debugger_config: dict = None,
        mode: str = "test",
    ) -> None:
        super().__init__(task_id=task_id)
        self.task_id = task_id
        self.task_list = None
        self.data_path = data_path
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
        return f"EHRGym.biocoder.{formatted_name}"
    
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
        self.question = task_data['problem']
        self.answer = validate_code(task_data['code'])["env_message"]
        self.code_id = task_data['idx']
        self.context = task_data['context']
        self.signature = task_data['signature']
        self.context_pattern = "<<insert solution here>>"
        goal, info = self.setup_goal()
        return goal, info

    def setup_goal(self) -> tuple[str, dict]:
        """
        Set up the goal and info for the task

        Parameters:
        -----------------
        data_path: str
            Path to the data directory
        """
        self.goal = self.question
        info = {
            "code_id": self.code_id,
        }
        return self.goal, info

    def _get_obs(self) -> dict:
        obs = {}
        obs["type"] = "initial_observation"
        obs["info"] = {}
        obs["info"]["question"] = self.question
        obs["info"]["code_id"] = self.code_id
        obs["info"]["task_goal"] = self.goal
        obs["info"]["context_info"] = context_information.format(context=self.context)
        obs["info"]["signature_info"] = signature_information.format(signature=self.signature)
        obs["info"]["instruction"] = instruction.format(signature_info=obs["info"]["signature_info"])

        return obs

    def validate(self, chat_messages, obs):
        if obs["type"] == "code_execution":
            pred = obs['env_message']
            correctness = self.answer == pred
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
