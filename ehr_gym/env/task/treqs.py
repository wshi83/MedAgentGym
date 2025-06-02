import os
from .base import AbstractEHRTask
import json

overall_information = """Assume you have knowledge of several tables:
(1) Tables are linked by identifiers which usually have the suffix 'ID'. For example, SUBJECT_ID refers to a unique patient. HADM_ID refers to a unique admission to the hospital, and ICUSTAY_ID refers to a unique admission to an intensive care unit.
(2) All tables contain SUBJECT_ID (patient identifier) and HADM_ID (hospital admission identifier).
(3) The table names are self-explanatory.
"""

table_information = """For different tables, they contain the following information:
(1) DEMOGRAPHIC.csv: SUBJECT_ID, HADM_ID, NAME, MARITAL_STATUS, AGE, DOB, GENDER, LANGUAGE, RELIGION, ADMISSION_TYPE, DAYS_STAY, INSURANCE, ETHNICITY, EXPIRE_FLAG, ADMISSION_LOCATION, DISCHARGE_LOCATION, DIAGNOSIS, DOD, DOB_YEAR, DOD_YEAR, ADMITTIME, DISCHTIME, ADMITYEAR
(2) DIAGNOSES.csv: SUBJECT_ID, HADM_ID, ICD9_CODE, SHORT_TITLE, LONG_TITLE
(3) LAB.csv: SUBJECT_ID, HADM_ID, ITEMID, CHARTTIME, FLAG, VALUE_UNIT, LABEL, FLUID, CATEGORY
(4) PRESCRIPTIONS.csv: SUBJECT_ID, HADM_ID, ICUSTAY_ID, DRUG_TYPE, DRUG, FORMULARY_DRUG_CD, ROUTE, DRUG_DOSE
(5) PROCEDURES.csv: SUBJECT_ID, HADM_ID, ICD9_CODE, SHORT_TITLE, LONG_TITLE

All the tabls are saved in the data directory {}.
"""

instruction = """You are an biomedical expert in handling EHR data and answer questions accordingly. 
Your objective is to solve a coding problem with given EHR data, with the goal of finally give a concrete answer to the question.
{overall}

{EHR_tables}
"""


class TreqsEHRTask(AbstractEHRTask):
    """
    Generic task for answering questions based on the Treqs EHR data.

    Class sttributed:
    -----------------
    config_path: str
        Path to the configuration file
    
    Parameters:
    -----------------
    task_id: int
        The id of the task inside the data.
    
    """
    permitted_actions = ['request_info', 'validate_code', 'debug']
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
        return f"EHRGym.treqs.{formatted_name}"
    
    def setup(self) -> tuple[str, dict]:
        """
        Set up the task

        Parameters:
        -----------------
        data_path: str
            Path to the data directory
        """
        file_list = os.listdir(self.data_path)
        # remain all the files end with .csv
        self.file_list = [file for file in file_list if file.endswith(".csv")]
        self.file_path_list = [os.path.join(self.data_path, file) for file in file_list]

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
        self.question = task_data['question']
        self.answer = task_data['answer']

        # locate the database
        self.database_directory = os.path.join(self.data_path, 'treqs.db')

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
        obs["info"]["table_info"] = table_information.format(self.data_path)
        obs["info"]["overall"] = overall_information
        obs["info"]["task_goal"] = self.goal
        obs["info"]["instruction"] = instruction.format(overall=overall_information, EHR_tables=table_information.format(self.data_path))
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
            pred = obs['env_message']
            if type(self.answer) == list:
                ans = self.answer[0]
            else:
                ans = self.answer
            # ans = self.answer
            old_flag = True
            if not ans in pred:
                old_flag = False
            if "True" in pred:
                pred = pred.replace("True", "1")
            else:
                pred = pred.replace("False", "0")
            if ans == "False" or ans == "false":
                ans = "0"
            if ans == "True" or ans == "true":
                ans = "1"
            if ans == "None" or ans == "none":
                ans = "0"
            if ", " in ans:
                ans = ans.split(', ')
            if ans[-2:] == ".0":
                ans = ans[:-2]
            if not type(ans) == list:
                ans = [ans]
            new_flag = True
            for i in range(len(ans)):
                if not ans[i] in pred:
                    new_flag = False
                    break
            correctness = old_flag or new_flag
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
    