import abc
import logging
from copy import copy, deepcopy
from dataclasses import asdict

class EHRPrompt:
    def __init__(self):
        self.prompt_template = """{instruction}
You have access to exactly three different actions with params, and receive corresponding feedback after each action:
1. request_info: Retrieve specific data information (column information or location of cells) from the EHR data
    - params: data_path (str), info_type (str: "column_names"/"column_values"/"term"), keyterm (str)
    - feedback: information you requested, the function is ONLY used to request information from EHR data.
2. validate_code: Test code execution to check the intermediate results or for final answer
    - params: code (str)
    - feedback: execution result (success or failure), error message if failed, code output if success
3. debug: Debug the code with the execution error message to find the problem
    - params: code (str), error_msg (str)
    - feedback: debugged code output (str)

Code requirements:
    - Request all information first.
    - Use the variable 'answer' to store the answer of the code.
    - Code should be self-contained and not rely on any variables or state outside.
    
Response format requirements, strictly one of the following:
{format_output}
    - Must be valid JSON format
    - No additional text or formatting allowed"""
        self.format_output = """{
    "action": "request_info",
    "params": {
        "data_path": "<data_path>",
        "info_type": "<info_type>",
        "keyterm": "<keyterm>"
    }
}
or
{
    "action": "validate_code",
    "params": {
        "code": "<code>"
    }
}
or
{
    "action": "debug",
    "params": {
        "code": "<code>",
        "error_msg": "<error_message>"
    }
}"""

class DynamicPrompt:
    def __init__(self):
        self.prompt_template = """{instruction}
You have access to exactly three different actions with params, and receive corresponding feedback after each action:
{action_definition}

Code requirements:
    - Request all information first.
    - Use the variable 'answer' to store the answer of the code.
    - Code should be self-contained and not rely on any variables or state outside.
    
Response format requirements, strictly one of the following:
{format_output}
    - Must be valid JSON format
    - No additional text or formatting allowed"""
        self.action_definition = {
            "request_info": """request_info: Retrieve specific data information (column information or location of cells) from the EHR data
    - params: data_path (str), info_type (str: "column_names"/"column_values"/"term"), keyterm (str)
    - feedback: information you requested, the function is ONLY used to request information from EHR data.""",
            "validate_code": """validate_code: Test code execution to check the intermediate results or for final answer
    - params: code (str)
    - feedback: execution result (success or failure), error message if failed, code output if success""",
            "debug": """debug: Debug the code with the execution error message to find the problem
    - params: code (str), error_msg (str)
    - feedback: debugged code output (str)""",
            "terminal": """terminal: Write terminal commands to install some mandatory packages or libraries for code execution
    - params: cmd (str)
    - feedback: execution result (success or failure), error message if failed, command output if success""",  
        }
        self.action_format = {
            "request_info": """{
    "action": "request_info",
    "params": {
        "data_path": "<data_path>",
        "info_type": "<info_type>",
        "keyterm": "<keyterm>"
    }
}""",
            "validate_code": """{
    "action": "validate_code",
    "params": {
        "code": "<code>"
    }
}""",
            "debug": """{
    "action": "debug",
    "params": {
        "code": "<code>",
        "error_msg": "<error_message>"
    }
}""",
            "terminal": """{
    "action": "terminal",
    "params": {
        "cmd": "<cmd>"
    }
}"""
        }