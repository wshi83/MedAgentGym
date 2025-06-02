import requests
import json
from datetime import datetime, timedelta

import os
from .base import AbstractEHRTask


fhir_overall_information = """In FHIR, there are a few common HTTP GET or POST requests to interact with the server. The descriptions of requests are listed here: {}. 
"""


class MedAgentBenchTask(AbstractEHRTask):
    """
    Generic task for answering questions based on the MedAgentBench FHIR EHR.

    Class sttributed:
    -----------------
    config_path: str
        Path to the configuration file
    
    Parameters:
    -----------------
    task_id: int
        The id of the task inside the data.
    
    """
    permitted_actions = ["validate_code", "debug", "terminal"]
    def __init__(
        self,
        task_id: int,
        data_path: str = None,
        debugger_config: dict = None,
        mode: str = "test",
    ) -> None:
        super().__init__(task_id=task_id)
        self.task_id = task_id
        self.fhir_api_base = "http://localhost:8080/fhir/"
        self.task_list = None
        self.func_json_path = os.path.join(data_path, "funcs_v1.json")
        self.data_path = data_path
        self.debugger_config = debugger_config
        self.mode = mode

        self.evaluator = MedAgentBenchAnswerEvaluator(fhir_api_base=self.fhir_api_base)
        try:
            with open(self.func_json_path, "r") as file:

                fhir_function_descriptions = str(json.load(file)) # convert from list of dicts to str

        except Exception as e:
            print(f"File not found error: {e}")
        self.fhir_overall_information = fhir_overall_information.format(fhir_function_descriptions)
    
    @classmethod
    def get_task_id(cls):
        # Get the class name and remove the word 'Task' from the end if it exists
        class_name = cls.__name__.replace("Task", "")
        # Convert CamelCase to hyphen-separated format
        formatted_name = "".join(
            ["-" + c.lower() if c.isupper() else c for c in class_name]
        ).lstrip("-")
        return f"EHRGym.medagentbench.{formatted_name}"
    
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

        if task_data["context"]:
            self.question = task_data['instruction'] + task_data['context']
        else:
            self.question = task_data['instruction']
        
        if task_data['sol']:
            self.answer = task_data['sol']
        else:
            self.answer = None

        # configure the task
        goal, info = self.setup_goal()
        return goal, info

    
    def setup_goal(self) -> tuple[str, dict]:
        """
        Set up the goal for the task
        """
        super().setup_goal()
        # get the task configuration
        answer_format = """answer = {"GET": ["60","S2874099"], "POST": ["http://localhost:8080/fhir/Observation", "payload]}
The answers to the questions are listed in "GET" instead of the get commands, while the post url and payload are listed in "POST"."""
        self.goal = f"""You are an expert in using FHIR functions to assist medical professionals.
{self.fhir_overall_information}

You are given a question and a set of possible functions. 
Based on the question, you will need to write a python code to achieve the purpose. 
    
    1. Write a python script to invoke a GET function of the FHIR server, you MUST put it in the format of\nGET url?param_name1=param_value1&param_name2=param_value2...
    2. Write a python script to invoke a POST function of the FHIR server, you MUST put it in the format of\nPOST url\n[your payload data in JSON format]
    3. If you have got answers for all the questions and finished all the requested tasks, you MUST save the final answers in the format of {answer_format} (make sure the list is JSON loadable.)
    
You SHOULD NOT include any other text in the response.

Please write the python code and use the variable 'answer' to store the answer of the code.
Question: {self.question}\n. The FHIR server base URL is {self.fhir_api_base}. Do not directly write the GET and POST requests.
        """
        
        info = {}
        return self.goal, info

    def _get_obs(self) -> dict:

        obs = {}
        obs["type"] = "initial_observation"
        obs["info"] = {}
        obs["info"]["overall"] = self.fhir_overall_information
        obs["info"]["task_goal"] = self.goal
        obs["info"]["instruction"] = self.goal
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
            # try:
            try:
                pred = pred.replace("'", "\"")
                pred = json.loads(pred)
            except Exception as e:
                print(f"Error in loading the prediction: {e}")
                pass
            try:
                evaluator_result = self.evaluator.eval(self.task_list[self.task_id], pred)
            except Exception as e:
                return (
                        0,
                        False,
                        "The evaluation failed",
                        {"message": f"There is an error occurred when evaluating the task. \nError Message: {str(e)}"}
                    )

            if evaluator_result["Connection"] and evaluator_result["Eval"]:
                return (
                    1, 
                    True, 
                    "The answer is correct", 
                    {"message": "The question is correctly solved."}
                )
            elif evaluator_result["Connection"] and not evaluator_result["Eval"]:
                return (
                    0,
                    False,
                    "The answer is incorrect",
                    {"message": "The question is not correctly solved. Can you think about whether there might be some mistakes in the previous code?"}
                )
            else: # no server connection
                return (
                        0,
                        False,
                        "The server is not connected",
                        {"message": f"The connection to the FHIR server is broken, please check the server"}
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
    


class MedAgentBenchAnswerEvaluator():
    def __init__(self, fhir_api_base):
        self.fhir_api_base = fhir_api_base

        try:
            assert self._verify_fhir_server(fhir_api_base) == True
        except Exception as e:
            print("FHIR server not connected!")
        

    def eval(self, case_data, results):
        """
            results: {"GET": [], "POST": []}
        """

        if self._verify_fhir_server(self.fhir_api_base):
            task_id = case_data["id"].split('_')[0]
        
            grader_func = getattr(self, str(task_id))
            return {"Connection": True, "Eval": grader_func(case_data, results, self.fhir_api_base)}
        else:
            return {"Connection": False, "Eval": None}
        

    def _verify_fhir_server(self, fhir_api_base):
        """
        Verify connection to FHIR server. Returns True if everything is good
        """
        res = self._send_get_request(f'{fhir_api_base}metadata')
        if res.get('status_code', 0) != 200:
            return False
        return True

    def _send_get_request(self, url, params=None, headers=None):
        """
        Sends a GET HTTP request to the given URL.

        Args:
            url (str): The URL to send the GET request to.
            params (dict, optional): Query parameters to include in the request. Defaults to None.
            headers (dict, optional): HTTP headers to include in the request. Defaults to None.

        Returns:
            dict: A dictionary containing the response's status code and data.

        Raises:
            requests.exceptions.RequestException: If an error occurs during the request.
        """
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()  # Raises an HTTPError if the response code is 4xx or 5xx
            return {
                "status_code": response.status_code,
                "data": response.json() if response.headers.get('Content-Type') == 'application/json' else response.text
            }
        except Exception as e:
            return {"error": str(e)}


    def _extract_posts(self, results):
        posts = []
        for idx, i in enumerate(results.history):
            if (i.role == 'agent') and ('POST' in i.content):
                if (idx<len(results.history)) and ("POST request accepted" in results.history[idx+1].content):
                    try:
                        r = i.content
                        url = r.split('\n')[0][4:].strip()
                        payload = json.loads('\n'.join(r.split('\n')[1:]))
                        posts.append((url, payload))
                    except:
                        pass
        return posts

    def _check_has_post(self, results):
        for i in results.history:
            if (i.role == 'agent') and ('POST' in i.content):
                return True
        return False
    
        
    def _calculate_age(self, dob):
        today = datetime(2023,11,13)
        # Calculate the difference in years
        age = today.year - dob.year
        # Adjust if the birthday hasn't occurred yet this year
        if (today.month, today.day) < (dob.month, dob.day):
            age -= 1
        return age

    def task1(self, case_data, results, fhir_api_base):

        """
            Example question: What\u2019s the MRN of the patient with name Maria Alvarez and DOB of 1940-03-05? If the patient does not exist, the answer should be \"Patient not found\"
        """
        # if self._check_has_post(results) is True: #Should not have any POST request
        #     return False
        ref_sol = case_data['sol'][0]
        try:
            if ref_sol in str(results["GET"]):
                return True
            return False
        except:
            return False


    def task2(self, case_data, results, fhir_api_base):
        """
            Example question: What's the age of the patient with MRN of S6538722? It's 2023-11-13T10:15:00+00:00 now, and the answer should be rounded down to an integer.
        """
        # if self._check_has_post(results) is True: #Should not have any POST request
        #     return False
        url = f"{fhir_api_base}Patient?identifier={case_data['eval_MRN']}&_format=json"
        get_res = json.loads(self._send_get_request(url)['data'])
        dob_str = get_res['entry'][0]['resource']['birthDate']
        parsed_date = datetime.strptime(dob_str, "%Y-%m-%d")
        ref_sol = str(self._calculate_age(parsed_date))
        try:

            if ref_sol in str(results["GET"]):
                return True
            return False
        except:
            return False


    def task3(self, case_data, results, fhir_api_base):
        """
            Example question: I just measured the blood pressure for patient with MRN of S1353305, and it is \"118\/77 mmHg\". Help me record it. It's 2023-11-13T10:15:00+00:00 now. The flowsheet ID for blood pressure is BP.
        """
        try:
            assert f'{fhir_api_base}Observation' in str(results)
            assert "resourceType" in str(results) and "Observation" in str(results)
            assert "http://hl7.org/fhir/observation-category" in str(results) and "vital-signs" in str(results) and "Vital Signs" in str(results)
            assert "text" in str(results) and "BP" in str(results)
            assert "effectiveDateTime" in str(results) and "2023-11-13T10:15:00+00:00" in str(results)
            assert "status" in str(results) and "final" in str(results)
            assert "valueString" in str(results) and "118/77 mmHg" in str(results)
            assert "reference" in str(results) and "Patient/S2380121" in str(results)
        except Exception as e:
            print(e, flush=True)
            return False
        return True


    def task4(self, case_data, results, fhir_api_base):
        """
            Example question: What\u2019s the most recent magnesium level of the patient S6330912 within last 24 hours? It's 2023-11-13T10:15:00+00:00 now. The code for magnesium is \"MG\". The answer should be a single number converted to a unit of mg\/dL, and it should be -1 if a measurement within last 24 hours is not available.
        """
        # if self._check_has_post(results) is True: #Should not have any POST request
        #     return False
        url = f"{fhir_api_base}Observation?patient={case_data['eval_MRN']}&code=MG&_count=5000&_format=json"
        get_res = json.loads(self._send_get_request(url)['data'])
        cutoff = datetime.fromisoformat("2023-11-13T10:15:00+00:00")
        last_meas, last_value = None, None
        for i in get_res.get('entry', []):
            effective_time = datetime.fromisoformat(i['resource']['effectiveDateTime'])
            value = i['resource']['valueQuantity']['value']
            if effective_time >= (cutoff - timedelta(hours=24)):
                if (last_meas is None) or (effective_time > last_meas):
                    last_meas = effective_time
                    last_value = value
        ref_sol = [last_value if last_value is not None else -1][0]
        # print(ref_sol)
        # print(case_data['id'], ref_sol, results, flush=True)
        try:
            if str(ref_sol) in str(results):
                return True
            return False
        except:
            return False

    def task5(self, case_data, results, fhir_api_base):
        """
            Example question: Check patient S1715871's last serum magnesium level within last 24 hours. If low, then order replacement IV magnesium according to dosing instructions. If no magnesium level has been recorded in the last 24 hours, don't order anything. It's 2023-11-13T10:15:00+00:00 now. The code for magnesium is \"MG\". The NDC for replacement IV magnesium is 0338-1715-40....
        """
        url = f"{fhir_api_base}Observation?patient={case_data['eval_MRN']}&code=MG&_count=5000&_format=json"
        get_res = json.loads(self._send_get_request(url)['data'])
        cutoff = datetime.fromisoformat("2023-11-13T10:15:00+00:00")
        last_meas, last_value = None, None
        for i in get_res.get('entry', []):
            effective_time = datetime.fromisoformat(i['resource']['effectiveDateTime'])
            value = i['resource']['valueQuantity']['value']
            if effective_time >= (cutoff - timedelta(hours=24)):
                if (last_meas is None) or (effective_time > last_meas):
                    last_meas = effective_time
                    last_value = value
        try:
            assert f'{fhir_api_base}MedicationRequest' in str(results)
            assert 'resourceType' in str(results) and 'MedicationRequest' in str(results)
            assert "http://hl7.org/fhir/sid/ndc" in str(results)
            assert "0338-1715-40" in str(results)
            assert 'authoredOn' in str(results) and '2023-11-13T10:15' in str(results)
            assert 'dosageInstruction' in str(results) and 'route' in str(results) and 'IV' in str(results)
            if last_value<1:
                dose, rate = 4, 4
            elif last_value<1.5:
                dose, rate = 2, 2
            else:
                dose, rate = 1, 1
            assert 'dosageInstruction' in str(results) and 'doseAndRate' in str(results) and 'doseQuantity' in str(results) and 'value' in str(results) and str(dose) in str(results) and 'unit' in str(results) and 'g' in str(results)
            assert 'dosageInstruction' in str(results) and 'doseAndRate' in str(results) and 'rateQuantity' in str(results) and 'value' in str(results) and str(rate) in str(results) and 'unit' in str(results) and 'h' in str(results)
            assert 'status' in str(results) and 'active' in str(results)
            assert 'intent' in str(results) and 'order' in str(results)
            assert 'subject' in str(results) and 'reference' in str(results) and f"Patient/{case_data['eval_MRN']}" in str(results)
        except Exception as e:
            print(e, flush=True)
            return False
        
        ref_sol = [last_value if last_value is not None else -1][0]
        # print(case_data['id'], ref_sol, results.result, flush=True)
        try:
            if (str(ref_sol) == str(results)) or ('[]' in str(results)): #We only ask the model to check, so it's fine if model returns []
                return True
            return False
        except:
            return False

    def task6(self, case_data, results, fhir_api_base):
        """
            Example question: What is the average CBG of the patient S6540602 over the last 24 hours? It's 2023-11-13T10:15:00+00:00 now. The code for CBG is \"GLU\". The answer should be a single number converted to a unit of mg\/dL, and it should be -1 if a measurement within last 24 hours is not available.
        """
        # if self._check_has_post(results) is True: #Should not have any POST request
        #     return False
        url = f"{fhir_api_base}Observation?patient={case_data['eval_MRN']}&code=GLU&_count=5000&_format=json"
        get_res = json.loads(self._send_get_request(url)['data'])
        cutoff = datetime.fromisoformat("2023-11-13T10:15:00+00:00")
        glu_sum, glu_count = 0., 0.
        for i in get_res.get('entry', []):
            effective_time = datetime.fromisoformat(i['resource']['effectiveDateTime'])
            value = i['resource']['valueQuantity']['value']
            if effective_time >= (cutoff - timedelta(hours=24)):
                glu_sum += value
                glu_count += 1
        
        ref_sol = [glu_sum/glu_count if glu_count != 0 else -1][0]
        print(ref_sol)
        # print(case_data['id'], ref_sol, results.result, flush=True)
        try:
            # l = results["GET"][0]
            if str(ref_sol) in str(results):
                return True
            return False
        except:
            return False

    def task7(self, case_data, results, fhir_api_base):
        """
            Example question: What is the most recent CBG of the patient S6530532? It's 2023-11-13T10:15:00+00:00 now. The code for CBG is \"GLU\
        """
        # if self._check_has_post(results) is True: #Should not have any POST request
        #     return False
        url = f"{fhir_api_base}Observation?patient={case_data['eval_MRN']}&code=GLU&_count=5000&_format=json"
        get_res = json.loads(self._send_get_request(url)['data'])
        last_meas, last_value = None, None
        for i in get_res.get('entry', []):
            effective_time = datetime.fromisoformat(i['resource']['effectiveDateTime'])
            value = i['resource']['valueQuantity']['value']
            if (last_meas is None) or (effective_time > last_meas):
                last_meas = effective_time
                last_value = value
        ref_sol = [last_value if last_value is not None else -1][0]

        # print(case_data['id'], ref_sol, results.result, flush=True)
        try:
            if str(ref_sol) in str(results):
                return True
            return False
        except:
            return False


    def task8(self, case_data, results, fhir_api_base):
        """
            Order orthopedic surgery referral for patient S3241217. Specify within the free text of the referral, \"Situation: acute left knee injury, Background: radiology report indicates ACL tear. Assessment: ACL tear grade II. Recommendation: request for Orthopedic service to evaluate and provide management recommendations. It's 2023-11-13T10:15:00+00:00 now. The SNOMED code for orthopedic surgery referral is 306181000000106.
        """
        comment = "Situation: acute left knee injury, Background: radiology report indicates ACL tear. Assessment: ACL tear grade II. Recommendation: request for Orthopedic service to evaluate and provide management recommendations."
        try:
            assert f'{fhir_api_base}ServiceRequest' in str(results)
            assert 'resourceType' in str(results) and 'ServiceRequest' in str(results)

            assert 'http://snomed.info/sct' in str(results)
            assert '306181000000106' in str(results)
            assert '2023-11-13T10:15:00+00:00' in str(results)
            assert 'status' in str(results) and 'active' in str(results)
            assert 'intent' in str(results) and 'order' in str(results)
            assert 'priority' in str(results) and 'stat' in str(results)
            assert comment in str(results) and 'note' in str(results) and 'text' in str(results)
            assert 'reference' in str(results) and f"Patient/{case_data['eval_MRN']}" in str(results)
        except Exception as e:
            print(e, flush=True)
            return False
        return True
    

    def task9(self, case_data, results, fhir_api_base):
        """
            Example question: Check patient S1311412's most recent potassium level. If low, then order replacement potassium according to dosing instructions. Also pair this order with a morning serum potassium level to be completed the next day at 8am. It's 2023-11-13T10:15:00+00:00 now. The code for potassium is \"K\"...
        """
        url = f"{fhir_api_base}Observation?patient={case_data['eval_MRN']}&code=K&_count=5000&_format=json"
        get_res = json.loads(self._send_get_request(url)['data'])
        cutoff = datetime.fromisoformat("2023-11-13T10:15:00+00:00")
        last_meas, last_value = None, None
        for i in get_res.get('entry', []):
            effective_time = datetime.fromisoformat(i['resource']['effectiveDateTime'])
            value = i['resource']['valueQuantity']['value']
            if (last_meas is None) or (effective_time > last_meas):
                last_meas = effective_time
                last_value = value
        try:
            assert f'{fhir_api_base}MedicationRequest' in str(results)
            assert 'resourceType' in str(results) and 'MedicationRequest' in str(results)
            assert 'medicationCodeableConcept' in str(results) and 'coding' in str(results) and 'system' in str(results) and "http://hl7.org/fhir/sid/ndc" in str(results)
            assert  "40032-917-01" in str(results)
            assert '2023-11-13T10:15'  in str(results)
            assert 'dosageInstruction' in str(results) and 'route' in str(results) and 'oral' in str(results).lower()
            dose = (3.5-last_value)/0.1 * 10
            assert 'dosageInstruction' in str(results) and 'doseAndRate' in str(results) and 'doseQuantity' in str(results) and 'value' in str(results) and str(dose) in str(results)
            assert 'dosageInstruction' in str(results) and 'doseAndRate' in str(results) and 'doseQuantity' in str(results) and 'unit' in str(results) and 'mEq' in str(results)
            assert 'status' in str(results) and 'active' in str(results)
            assert 'intent' in str(results) and 'order' in str(results)
            assert 'subject' in str(results) and 'reference' in str(results) and f"Patient/{case_data['eval_MRN']}" in str(results)
        except Exception as e:
            print(e, flush=True)
            return False
        
        try:
            assert f'{fhir_api_base}ServiceRequest' in str(results)
            assert 'resourceType' in str(results) and 'ServiceRequest' in str(results)
            assert 'http://loinc.org' in str(results)
            assert '2823-3' in str(results)
            assert '2023-11-13T10:15:00+00:00' in str(results)
            assert 'status' in str(results) and 'active' in str(results)
            assert 'intent' in str(results) and 'stat' in str(results)
            assert 'subject' in str(results) and 'reference' in str(results) and f"Patient/{case_data['eval_MRN']}" in str(results)
            assert '2023-11-14T08:' in str(results)
        except Exception as e:
            print(e, flush=True)
            return False

        ref_sol = [last_value if last_value is not None else -1][0]
        # print(case_data['id'], ref_sol, results.result, flush=True)
        try:
            if (str(ref_sol) in str(results)) or (str([]) in str(results)): #We only ask the model to check, so it's fine if model returns []
                return True
            return False
        except:
            return False


    def task10(self, case_data, results, fhir_api_base):
        """
            Example question: What\u2019s the last HbA1C (hemoglobin A1C) value in the chart for patient S2016972 and when was it recorded? If the lab value result date is greater than 1 year old, order a new HbA1C lab test. It's 2023-11-13T10:15:00+00:00 now. The code for HbA1C is \"A1C\". The LOINC code for ordering HbA1C lab is: 4548-4. T
        """
        url = f"{fhir_api_base}Observation?patient={case_data['eval_MRN']}&code=A1C&_count=5000&_format=json"
        get_res = json.loads(self._send_get_request(url)['data'])
        cutoff = datetime.fromisoformat("2023-11-13T10:15:00+00:00")
        last_meas, last_value, last_time = None, None, None
        for i in get_res.get('entry', []):
            effective_time = datetime.fromisoformat(i['resource']['effectiveDateTime'])
            value = i['resource']['valueQuantity']['value']
            if (last_meas is None) or (effective_time > last_meas):
                last_meas = effective_time
                last_time = i['resource']['effectiveDateTime']
                last_value = value
        
        if (last_value is None):
            ref_sol = -1
        else: 
            ref_sol = last_value
        
        if (last_value is None) or (last_meas < datetime.fromisoformat("2022-11-13T10:15:00+00:00")): #Order needed
            # posts = self._extract_posts(results)
            # if len(posts) != 1: #Should be one for A1C test
            #     return False
            try:
                assert f'{fhir_api_base}ServiceRequest' in str(results)
                assert 'resourceType' in str(results) and 'ServiceRequest' in str(results)
                assert 'http://loinc.org' in str(results)
                assert '4548-4' in str(results)
                assert 'authoredOn' in str(results) and '2023-11-13T10:15:00+00:00' in str(results)
                assert 'status' in str(results) and 'active' in str(results)
                assert 'intent' in str(results) and 'order' in str(results)
                assert 'priority' in str(results) and 'stat' in str(results)
                assert 'subject' in str(results) and 'reference' in str(results) and f"Patient/{case_data['eval_MRN']}" in str(results)
            except Exception as e:
                print(e, flush=True)
                return False
        else:#No order needed
            # if self._check_has_post(results) is True:
            #     return False
            if len(results["POST"]) != 0:
                return False


        # print(case_data['id'], ref_sol, results.result, flush=True)
        try:
            if (str(ref_sol) == str(results)) or (str([]) == str(results)): #We only ask the model to check, so it's fine if model returns []
                return True
            return False
        except:
            return False