import os
from .base import AbstractEHRTask
import json
import pandas as pd
import torch
import csv

instruction = """You are an biomedical expert in writing machine learning code to solve EHR-relevant tasks.
Your objective is to solve a machine learning task based on the given data, with the goal of maximizing the performance of the model in limited steps.
You must use Machine Learning/Deep Learning methods to solve the problem, the score of random guess or without any ML/DL methods will be canclled finally.
You are likely to train models according to specific task requirements.
You have access to a GPU and several CPUs for training DL/ML models.
Use CUDA and PyTorch for faster training if needed.

Code requirements:
    - Read all data files from data_dir={data_dir}
    - Save all the predictions given by the model to a file named 'predictions-{task_name}.csv' in the './cache/ehrshot/{model}/' directory.
    - Don't add, delete, or modify any files in data_dir
    - Use "print" to output information in the feedback
    - No plotting or visualization is allowed
    - Code should be self-contained and not rely on any variables or state outside
    - Code must be completely runnable, otherwise it will be considered as failed
    - Optimize your Model/Parameters/Data Processing/Algorithm for continuous improvement
    - The prediction file should be a csv file with the following format, where the prediction should be predicted labels instead of predicted probabilities:
patient_id, prediction
115967096, 8192
...

{feature_information}

{label_information}
"""

feature_information = """The corresponding features are stored in the following directories:
{feature_directory_train}: training features for the task
{feature_directory_val}: validation features for the task
{feature_directory_test}: test features for the task
Each of the feature files is a dictionary, containing the following keys:
    - data_matrix: the feature vectors of the visits, where each row is a embedded vector, representing a single visit of a patient
    - patient_ids: the identifiers of the patients, where each row is a visit and the corresponding patient id
    - labeling_time: the time of the visit, where each row is a visit and the corresponding time
"""

label_information = """The corresponding labels are stored in the following directories:
{label_directory_train}: training labels for the task
{label_directory_val}: validation labels for the task
{label_directory_test}: test labels for the task
Each of the label files contain the following columns:
    - patient_id: the identifier of the patient
    - value: the label value of the patient on the {task_name} task
    - label_type: the type of the label, which can be 'categorical'/'boolean', etc.
    - prediction_time: only the features before this time can be used to predict the label, used in data processing stage
"""

class EHRShotTask(AbstractEHRTask):
    """
    Generic task for answering questions based on the EHRShot data.

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
        self.output_directory = './cache/ehrshot/{}'.format(debugger_config["model_name"])
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        self.mode = mode
    @classmethod
    def get_task_id(cls):
        # Get the class name and remove the word 'Task' from the end if it exists
        class_name = cls.__name__.replace("Task", "")
        # Convert CamelCase to hyphen-separated format
        formatted_name = "".join(
            ["-" + c.lower() if c.isupper() else c for c in class_name]
        ).lstrip("-")
        return f"EHRGym.ehrshot.{formatted_name}"

    def setup(self) -> tuple[str, dict]:
        """
        Set up the task.

        Returns:
            tuple[str, dict]: A tuple containing the task description and a dictionary of task parameters.
        """
        train_data_path = os.path.join(self.data_path, "clmbr_train.pkl")
        val_data_path = os.path.join(self.data_path, "clmbr_val.pkl")
        test_data_path = os.path.join(self.data_path, "clmbr_test.pkl")

        if self.task_list is None:
            task_file = 'task.jsonl'
            task_path = os.path.join(self.data_path, task_file)
            self.task_list = []
            with open(task_path, 'r') as f:
                for line in f:
                    self.task_list.append(json.loads(line))
        task_data = self.task_list[self.task_id]
        self.question = task_data['task_description']
        self.task_name = task_data['task_name']
        self.label_directory = os.path.join(self.data_path, self.task_name)
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
            "task_name": self.task_name,
            "feature_directory_train": os.path.join(self.data_path, "clmbr_train.pkl"),
            "feature_directory_val": os.path.join(self.data_path, "clmbr_val.pkl"),
            "feature_directory_test": os.path.join(self.data_path, "clmbr_test.pkl"),
            "label_directory_train": os.path.join(self.label_directory, "train_labels.csv"),
            "label_directory_val": os.path.join(self.label_directory, "val_labels.csv"),
            "label_directory_test": os.path.join(self.label_directory, "test_labels.csv"),
            "output_directory": self.output_directory,
        }
        return self.goal, info
    
    def _get_obs(self) -> dict:
        obs = {}
        obs["type"] = "initial_observation"
        obs["info"] = {}
        obs["info"]["question"] = self.question
        obs["info"]["task_name"] = self.task_name
        obs["info"]["task_goal"] = self.goal
        obs["info"]["feature_information"] = feature_information.format(
            feature_directory_train=os.path.join(self.data_path, "clmbr_train.pkl"),
            feature_directory_val=os.path.join(self.data_path, "clmbr_val.pkl"),
            feature_directory_test=os.path.join(self.data_path, "clmbr_test.pkl"),
        )
        obs["info"]["label_information"] = label_information.format(
            label_directory_train=os.path.join(self.label_directory, "train_labels.csv"),
            label_directory_val=os.path.join(self.label_directory, "val_labels.csv"),
            label_directory_test=os.path.join(self.label_directory, "test_labels.csv"),
            task_name=self.task_name,
        )
        obs["info"]["instruction"] = instruction.format(
            data_dir=self.data_path,
            task_name=self.task_name,
            model=self.debugger_config["model_name"],
            feature_information=obs["info"]["feature_information"],
            label_information=obs["info"]["label_information"],
        )
        return obs

    def validate(self, chat_messages, obs):
        if obs["type"] == "code_execution":
            pred = obs["env_message"]
            if obs["status"] == "SUCCESS":
                prediction_file = os.path.join(self.output_directory, "predictions-{}.csv".format(self.task_name))
                ground_truth_file = os.path.join(self.label_directory, "test_labels.csv")
                if os.path.exists(prediction_file):
                    # compute the accuracy between the prediction and the ground truth
                    try:
                        with open(ground_truth_file, 'r') as f:
                            reader = csv.DictReader(f)
                            ground_truth = {}
                            for row in reader:
                                pid = row['patient_id']
                                if 'tensor' in pid:
                                    pid = int(pid.split('(')[-1].split(')')[0])
                                else:
                                    pid = int(pid)
                                ground_truth[pid] = row['value']
                        with open(prediction_file, 'r') as f:
                            reader = csv.DictReader(f)
                            predictions = {}
                            for row in reader:
                                pid = row['patient_id']
                                if 'tensor' in pid:
                                    pid = int(pid.split('(')[-1].split(')')[0])
                                else:
                                    pid = int(pid)
                                predictions[pid] = row['prediction']
                        accuracy = 0
                        patient_id_list = list(predictions.keys())
                        notfound = 0
                        for patient_id in patient_id_list:
                            try:
                                prediction = predictions[patient_id]
                                ground_truth_value = ground_truth[patient_id]
                                if ground_truth_value == 'False':
                                    ground_truth_value = '0'
                                elif ground_truth_value == 'True':
                                    ground_truth_value = '1'
                                if prediction == 'False':
                                    prediction = '0'
                                elif prediction == 'True':
                                    prediction = '1'
                                if int(prediction) == int(ground_truth_value):
                                    accuracy += 1
                                else:
                                    # print("Patient ID: ", patient_id)
                                    # print("Prediction: ", prediction)
                                    # print("Ground Truth: ", ground_truth_value)
                                    # print("===================================")
                                    # input()
                                    continue
                            except Exception as e:
                                # print(e)
                                notfound += 1
                                continue
                        accuracy = accuracy / len(predictions)
                    except Exception as e:
                        error_msg = f"Error in evaluating the generated prediction file and the ground-truth file: {e}"
                        print(error_msg)
                        return (
                            0,
                            True,
                            error_msg,
                            {"message": error_msg}
                        )
                    return (
                        accuracy,
                        True,
                        "Successfully write machine learning code to solve the task. The accuracy is {:.2f}.".format(accuracy),
                        {"message": "The question is correctly solved. The accuracy is {:.2f}.".format(accuracy)},
                    )
                else:
                    return (
                        0,
                        False,
                        "Failed to write machine learning code to solve the task. The prediction file is not found.",
                        {"message": "The question is not correctly solved. The prediction file is not found."},
                    )
            else:
                return (
                    0,
                    False,
                    "Failed to write machine learning code to solve the task. The code execution failed.",
                    {"message": "The question is not correctly solved. The code execution failed."},
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
    



