import os
from .base import AbstractEHRTask
import json
import pandas as pd
import torch
from collections import defaultdict
import csv

instruction = """You are an biomedical expert in writing machine learning code to solve EHR-relevant tasks.
Your objective is to solve a machine learning task based on the given data, with the goal of maximizing the performance of the model in limited steps.
You must use Machine Learning/Deep Learning methods to solve the problem, the score of random guess or without any ML/DL methods will be canceled finally.
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


You have the data splits based on hospital admission ids. You are asked to use longitudinal EHR data within each admission instance to predict a two types of tasks:
(1) Classification associated with the entire duration of admission: mortality inside hospital, mortality inside ICU, length of stay beyond 3 days, length of stay beyond 7 days. All 4 are binary classification tasks using lab features only.
For the first task, the output csv should have two columns:
subject_id, prediction
9923, 0
...

(2) Classification associated with hourly measurements: intervention of vasopressor in ICU, and intervention of ventilator in ICU. Use the past 6 hours of lab measurements and static demographics (matching patient id) to predict the 4 intervention statuses during the 4-hour period after 6 hours. 
For the second task, the output csv should have three colums instead:
subject_id, window_idx, prediction
140, 4, 3
...

{feature_information}

{label_information}
"""

lab_feature_information = """The corresponding features are stored in the following directories:
{feature_directory_train}: training features for the task
{feature_directory_val}: validation features for the task
{feature_directory_test}: test features for the task
Each of the feature files is a pickled pandas dataframe:
    - subject_id: the unique ID of the subject
    - hadm_id: the unique ID of the hospital admission 
    - icustay_id: the unique ID of the ICU session
    - hours_in: the number of hours since hospital admission. Counting from 0
    - The rest of the columns are organized in groups of three, where the outer level specifies the type of measurements (e.g. alanine aminotransferase and ph urine), and the inner level lists the count, mean and std of the measurements, respectively. The table has been imputed.
"""

static_feature_information = """The corresponding features are stored in the following directories:
{feature_directory_train}: demographic training features for the task
{feature_directory_val}: demographic validation features for the task
{feature_directory_test}: demographic test features for the task
Each of the feature files is a pickled pandas dataframe:
    - subject_id: the unique ID of the subject
    - hadm_id: the unique ID of the hospital admission
    - icustay_id: the unique ID of the ICU session
    - intime: the total number of hours in the associated admission
    - gender_F and gender_M: one-hot boolean columns for gender
    - Age 1.0, Age 2.0, Age 3.0, Age 4.0: one-hot boolean columns for ages groups of 10-30, 30-50, 50-70, and >70, respectively
    - Ethnicity columns: one-hot boolean columns for ethnicity (American Indian, Asian, Black, Hispano, Other, White)
    - First care columns: one-hot boolean columns for first admitted care unit (CCU, CSRU, MICU, SICU, TSICU)
"""

mort_los_label_information = """The corresponding labels are stored in the following directories:
{label_directory_train}: training labels for the task
{label_directory_val}: validation labels for the task
{label_directory_test}: test labels for the task
Each of the label csv files contain the following columns:
    - subject_id: the unique ID of the subject
    - hadm_id: the unique ID of the hospital admission
    - mort_icu or mort_hosp or los_3 or los_7: the boolean label for whether the patient died in the ICU, died in hospital, the length of stay exceeding 3 days, and LOS exceeding 7 days, respectively
    - label_type: the type of the label, which can be 'categorical'/'boolean', etc.
"""

ventilator_vasopressor_label_information = """The corresponding labels are stored in the following directories:
{label_directory_train}: training labels for the task
{label_directory_val}: validation labels for the task
{label_directory_test}: test labels for the task
Each of the label csv files contain the following columns:
    - subject_id: the unique ID of the subject
    - 6_hour_window_id: the 6 hour predicted window counted since the patient is admitted to hospital.
    - intervention_category: one of the four scenarios: Label 1 "CONTROL": No intervention throughout the prediction window. Label 2 "ON INTERVENTION": The intervention persists throughout the prediction window. Label 3 "ONSET": Intervention starts within the prediction window. Label 4 "WEAN": Intervention ends within the prediction window.
    - label_type: the type of the label, which can be 'categorical'/'boolean', etc.
"""


class MIMICEXTRACTEHRTask(AbstractEHRTask):
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
        self.output_directory = './cache/mimic-extract/{}'.format(debugger_config["model_name"])
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
        return f"EHRGym.mimic-extract.{formatted_name}"

    def setup(self) -> tuple[str, dict]:
        """
        Set up the task.

        Returns:
            tuple[str, dict]: A tuple containing the task description and a dictionary of task parameters.
        """

        if self.task_list is None:
            task_file = 'test_tasks.jsonl'
            task_path = os.path.join(self.data_path, task_file)
            self.task_list = []
            with open(task_path, 'r') as f:
                for line in f:
                    self.task_list.append(json.loads(line))
        task_data = self.task_list[self.task_id]
        self.question = task_data['task_description']
        self.task_name = task_data['task_name']
        self.feature_label_directory = os.path.join(self.data_path, task_data['feature_label_directory'].split('./')[-1])
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

        if self.task_name in ['mort_icu', 'mort_hosp', 'los_7', 'los_3']:
            info = {
                "task_name": self.task_name,
                "lab_feature_directory_train": os.path.join(self.feature_label_directory, "X_lab_train.pkl"),
                "lab_feature_directory_val": os.path.join(self.feature_label_directory, "cX_lab_val.pkl"),
                "lab_feature_directory_test": os.path.join(self.feature_label_directory, "X_lab_test.pkl"),
                "label_directory_train": os.path.join(self.feature_label_directory, f"train_{self.task_name}_labels.csv"),
                "label_directory_val": os.path.join(self.feature_label_directory, f"val_{self.task_name}_labels.csv"),
                "label_directory_test": os.path.join(self.feature_label_directory, f"test_{self.task_name}_labels.csv"),
                "output_directory": self.output_directory,
            }
        elif self.task_name in ['vasopressor', 'ventilation']:
            info = {
                "task_name": self.task_name,
                "lab_feature_directory_train": os.path.join(self.feature_label_directory, "X_lab_train.pkl"),
                "lab_feature_directory_val": os.path.join(self.feature_label_directory, "X_lab_val.pkl"),
                "lab_feature_directory_test": os.path.join(self.feature_label_directory, "X_lab_test.pkl"),
                "demographic_feature_directory_train": os.path.join(self.feature_label_directory, "X_static_train.pkl"),
                "demographic_feature_directory_val": os.path.join(self.feature_label_directory, "X_static_val.pkl"),
                "demographic_feature_directory_test": os.path.join(self.feature_label_directory, "X_static_test.pkl"),
                "label_directory_train": os.path.join(self.feature_label_directory, f"train_{self.task_name}_labels.csv"),
                "label_directory_val": os.path.join(self.feature_label_directory, f"val_{self.task_name}_labels.csv"),
                "label_directory_test": os.path.join(self.feature_label_directory, f"test_{self.task_name}_labels.csv"),
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

        if self.task_name in ['mort_icu', 'mort_hosp', 'los_7', 'los_3']:

            obs["info"]["feature_information"] = lab_feature_information.format(
                feature_directory_train=os.path.join(self.feature_label_directory, "X_lab_train.pkl"),
                feature_directory_val=os.path.join(self.feature_label_directory, "X_lab_val.pkl"),
                feature_directory_test=os.path.join(self.feature_label_directory, "X_lab_test.pkl"),
            )
            obs["info"]["label_information"] = mort_los_label_information.format(
                label_directory_train=os.path.join(self.feature_label_directory, f"train_{self.task_name}_labels.csv"),
                label_directory_val=os.path.join(self.feature_label_directory, f"val_{self.task_name}_labels.csv"),
                label_directory_test=os.path.join(self.feature_label_directory, f"test_{self.task_name}_labels.csv"),
                task_name=self.task_name,
            )
        elif self.task_name in ['vasopressor', 'ventilation']:

            obs["info"]["feature_information"] = lab_feature_information.format(
                feature_directory_train=os.path.join(self.feature_label_directory, "X_lab_train.pkl"),
                feature_directory_val=os.path.join(self.feature_label_directory, "X_lab_val.pkl"),
                feature_directory_test=os.path.join(self.feature_label_directory, "X_lab_test.pkl"),
            ) + static_feature_information.format(
                feature_directory_train=os.path.join(self.feature_label_directory, "X_static_train.pkl"),
                feature_directory_val=os.path.join(self.feature_label_directory, "X_static_val.pkl"),
                feature_directory_test=os.path.join(self.feature_label_directory, "X_static_test.pkl"),
            )
            obs["info"]["label_information"] = mort_los_label_information.format(
                label_directory_train=os.path.join(self.feature_label_directory, f"train_{self.task_name}_labels.csv"),
                label_directory_val=os.path.join(self.feature_label_directory, f"val_{self.task_name}_labels.csv"),
                label_directory_test=os.path.join(self.feature_label_directory, f"test_{self.task_name}_labels.csv"),
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
                ground_truth_file = os.path.join(self.feature_label_directory, f"test_{self.task_name}_labels.csv")
                if os.path.exists(prediction_file):
                    if self.task_name in ['mort_icu', 'mort_hosp', 'los_7', 'los_3']:
                        # label associated with each patient
                        # compute the accuracy between the prediction and the ground truth
                        try:
                            with open(ground_truth_file, 'r') as f:
                                reader = csv.DictReader(f)
                                ground_truth = {}
                                for row in reader:
                                    pid = row['subject_id']
                                    if 'tensor' in pid:
                                        pid = int(pid.split('(')[-1].split(')')[0])
                                    else:
                                        pid = int(pid)
                                    ground_truth[pid] = row[self.task_name] # col names los_7 etc 
                            with open(prediction_file, 'r') as f:
                                reader = csv.DictReader(f)
                                predictions = {}
                                for row in reader:
                                    pid = row['subject_id']
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
                    elif self.task_name in ['vasopressor', 'ventilation']:
                        # compute the accuracy between the prediction and the ground truth
                        try:
                            with open(ground_truth_file, 'r') as f:
                                reader = csv.DictReader(f)
                                ground_truth = defaultdict()  # dict of list
                                for row in reader:
                                    pid = row['subject_id']
                                    if 'tensor' in pid:
                                        pid = int(pid.split('(')[-1].split(')')[0])
                                    else:
                                        pid = int(pid)

                                    # a list of labels for each unique patient. i.e. will not be judged on the predicted window index
                                    ground_truth[pid].append(row['intervention_category'])   
                            with open(prediction_file, 'r') as f:
                                reader = csv.DictReader(f)
                                predictions = defaultdict()  # dict of list
                                for row in reader:
                                    pid = row['subject_id']
                                    if 'tensor' in pid:
                                        pid = int(pid.split('(')[-1].split(')')[0])
                                    else:
                                        pid = int(pid)
                                    predictions[pid].append(row['prediction'])
                            accuracy = 0
                            patient_id_list = list(predictions.keys())
                            notfound = 0
                            total_predicted_windows = 0
                            for patient_id in patient_id_list:
                                try:
                                    list_of_prediction = predictions[patient_id]
                                    list_of_ground_truth_value = ground_truth[patient_id]

                                    total_predicted_windows += len(list_of_ground_truth_value) 
                                    if len(list_of_prediction) != len(list_of_ground_truth_value): # do not have the same number of predicted windows
                                        continue
                                    
                                    for i in range(len(list_of_prediction)):
                                        prediction, ground_truth_value  = list_of_prediction[i], list_of_ground_truth_value[i]
                                    
                                        if int(prediction) == int(ground_truth_value): # label 0, 1, 2, 3
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