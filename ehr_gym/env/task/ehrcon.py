import os
from .base import AbstractEHRTask
import json

overall_information = """Assume you have knowledge of several tables:
(1) Tables are linked by identifiers which usually have the suffix 'ID'. For example, SUBJECT_ID refers to a unique patient, HADM_ID refers to a unique admission to the hospital, and ICUSTAY_ID refers to a unique admission to an intensive care unit.
(2) Charted events such as notes, laboratory tests, and fluid balance are stored in a series of 'events' tables. For example the outputevents table contains all measurements related to output for a given patient, while the labevents table contains laboratory test results for a patient.
(3) Tables prefixed with 'd_' are dictionary tables and provide definitions for identifiers. For example, every row of chartevents is associated with a single ITEMID which represents the concept measured, but it does not contain the actual name of the measurement. By joining chartevents and d_items on ITEMID, it is possible to identify the concept represented by a given ITEMID.
(4) For the databases, four of them are used to define and track patient stays: admissions, patients, icustays, and transfers. Another four tables are dictionaries for cross-referencing codes against their respective definitions: d_icd_diagnoses, d_icd_procedures, d_items, and d_labitems. The remaining tables, including chartevents, cost, inputevents_cv, labevents, microbiologyevents, outputevents, prescriptions, procedures_icd, contain data associated with patient care, such as physiological measurements, caregiver observations, and billing information.
"""

table_information = """For different tables, they contain the following information:
(1) ADMISSIONS.csv: ROW_ID, SUBJECT_ID, HADM_ID, ADMITTIME, DISCHTIME, ADMISSION_TYPE, ADMISSION_LOCATION, DISCHARGE_LOCATION, INSURANCE, LANGUAGE, MARITAL_STATUS, ETHNICITY, AGE
(2) CHARTEVENTS.csv: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, ITEMID, CHARTTIME, VALUENUM, VALUEUOM
(3) COST.csv: ROW_ID, SUBJECT_ID, HADM_ID, EVENT_TYPE, EVENT_ID, CHARGETIME, COST
(4) D_ICD_DIAGNOSES.csv: ROW_ID, ICD9_CODE, SHORT_TITLE, LONG_TITLE
(5) D_ICD_PROCEDURES.csv: ROW_ID, ICD9_CODE, SHORT_TITLE, LONG_TITLE
(6) D_ITEMS.csv: ROW_ID, ITEMID, LABEL, LINKSTO
(7) D_LABITEMS.csv: ROW_ID, ITEMID, LABEL
(8) DIAGNOSES_ICD.csv: ROW_ID, SUBJECT_ID, HADM_ID, ICD9_CODE, CHARTTIME
(9) ICUSTAYS.csv: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, FIRST_CAREUNIT, LAST_CAREUNIT, FIRST_WARDID, LAST_WARDID, INTIME, OUTTIME
(10) INPUTEVENTS_CV.csv: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, CHARTTIME, ITEMID, AMOUNT
(11) LABEVENTS.csv: ROW_ID, SUBJECT_ID, HADM_ID, ITEMID, CHARTTIME, VALUENUM, VALUEUOM
(12) MICROBIOLOGYEVENTS.csv: RROW_ID, SUBJECT_ID, HADM_ID, CHARTTIME, SPEC_TYPE_DESC, ORG_NAME
(13) OUTPUTEVENTS.csv: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, CHARTTIME, ITEMID, VALUE
(14) PATIENTS.csv: ROW_ID, SUBJECT_ID, GENDER, DOB, DOD
(15) PRESCRIPTIONS.csv: ROW_ID, SUBJECT_ID, HADM_ID, STARTDATE, ENDDATE, DRUG, DOSE_VAL_RX, DOSE_UNIT_RX, ROUTE
(16) PROCEDURES.csv: ROW_ID, SUBJECT_ID, HADM_ID, ICD9_CODE, CHARTTIME
(17) TRANSFERS.csv: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, EVENTTYPE, CAREUNIT, WARDID, INTIME, OUTTIME

All the tables are saved in the a .db file at {db_location}.

In addition, you have access to a csv containing the clinical notes with the matching subject ids and hospital admission ids: ROW_ID, SUBJECT_ID, HADM_ID, CHARTDATE, CHARTTIME, STORETIME, CATEGORY, DESCRIPTION, CGID, ISERROR, TEXT, ADMITTIME

This clinical note csv is at {note_csv}.

"""

instruction = """You are an biomedical expert in handling EHR data and answer questions accordingly. 
Your objective is to solve a coding problem with given EHR data, with the goal of finally give a concrete answer to the question.
{overall}

{EHR_tables}
"""


class EHRCONEHRTask(AbstractEHRTask):
    """
    Generic task for answering questions based on the EHRCon EHR data.

    Class sttributed:
    -----------------
    config_path: str
        Path to the configuration file
    
    Parameters:
    -----------------
    task_id: int
        The id of the task inside the data.
    
    """
    permitted_actions = ['request_info', 'validate_code', 'debug', 'terminal']
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
        self.note_csv_path = os.path.join(self.data_path, 'clinical_notes.csv')
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
        return f"EHRGym.ehr-con.{formatted_name}"
    
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

        self.question = f"""
            For a given admission id {task_data['Hospital admission']} and the corresponding medical entity {task_data['Medical entity']}, identify if there is any inconsistency between the database and the clinical note. Return the answer as a boolean value
        """

        self.answer = task_data['Has error'] #boolean

        # locate the database
        self.database_directory = os.path.join(self.data_path, 'EHRCon_combined.db')

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
        obs["info"]["table_info"] = table_information.format(db_location=self.database_directory, note_csv=self.note_csv_path)
        obs["info"]["overall"] = overall_information
        obs["info"]["task_goal"] = self.goal
        obs["info"]["instruction"] = instruction.format(overall=overall_information, EHR_tables=obs["info"]["table_info"])
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
            
            correctness = False

            if isinstance(pred, str):
                if 'true' in pred.lower(): pred = True
                elif 'false' in pred.lower(): pred = False
            
            if pred == self.answer:
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
    