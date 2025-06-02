import time
import os
from pathlib import Path
import json
from typing import Literal, Dict, Any, Optional, Union, List, Tuple, Type
from ehr_gym.utils.env_utils import parse_and_truncate_error
import pandas as pd
import Levenshtein
import traceback
import subprocess
from datetime import datetime
from tqdm import tqdm
import logging
import uuid
from ehr_gym.llm.chat_api import OpenAIModelArgs, AzureModelArgs, VLLMModelArgs, make_system_message, make_user_message, make_assistant_message

logging.basicConfig(
    level=logging.INFO, format="%(name)s : %(levelname)-8s : %(message)s"
)
logger = logging.getLogger(__name__)

send_message_to_user: callable = None
report_infeasible_instructions: callable = None
retry_with_force: bool = False
request_info: callable = None
validate_code: callable = None
debug: callable = None


def send_msg_to_user(text: str):
    """
    Sends a message to the user.

    Examples:
        send_msg_to_user("Based on the results of my search, the city was built in 1960.")
    """
    send_message_to_user(text)

def report_infeasible(text: str):
    """
    Sends a message to the user.

    Examples:
        report_infeasible("I cannot follow these isntructions because there is no email field in this form.")
    """
    report_infeasible_instructions(text)

# request the relevant information
def request_info(data_path: str, info_type: str, keyterm: str) -> str:
    """
    Request a specific type of information from the data.

    Examples:
        request_info("./mimic_iii/", "term", "aspirin")
    """
    results = {}
    if info_type == 'column_names':
        try:
            df = pd.read_csv(data_path)
            column_names = df.columns.tolist()
            results = "The column names are: {}".format(column_names)
        except:
            results = f"The file is not a CSV file or the file does not exist. The error message: {e}"
        return results
    elif info_type == 'column_values':
        try:
            df = pd.read_csv(data_path)
            if keyterm not in df.columns:
                results = "The column name {} does not exist in the file.".format(keyterm)
                return results
            column_values = list(set(df[keyterm].values.flatten()))
            if len(column_values) <= 10:
                results = "The column values include: {}".format(column_values)
            else:
                results = "Too many values in the column, please specify a more detailed keyterm with info_type='term'. It will return all the relevant information to one keyword in the entire database."
        except Exception as e:
            results = f"The file is not a CSV file or the file does not exist. The error message: {e}"
        return results
    if keyterm == "":
        results = "The keyterm is empty."
        return results
    if data_path.endswith(".csv"):
        # file_list = [data_path.split("/")[-1]]
        data_path = '/'.join(data_path.split("/")[:-1])
        file_list = os.listdir(data_path)
    else:
        file_list = os.listdir(data_path)
    for file in file_list:
        file_path = os.path.join(data_path, file)
        if file.endswith(".csv"):
            df = pd.read_csv(file_path)
            # ambiguous search for the keyterm in the dataframe
            # if the keyterm is not found in the cells of the dataframe, return 5 most similar values
            # step 1: get all the possible cell values in the dataframe
            all_values = df.values.flatten()
            if keyterm in all_values:
                # find which column contains the keyterm
                columns = []
                for col in df.columns:
                    if keyterm in df[col].values:
                        columns.append(col)
                results[file] = """The table contains the keyterm "{}" under the following columns {}.""".format(keyterm, columns)
            else:
                levenshtein_dist = {}
                for cv in all_values:
                    levenshtein_dist[cv] = Levenshtein.distance(str(cv), str(keyterm)) / len(str(keyterm))
                levenshtein_dist = sorted(levenshtein_dist.items(), key=lambda x: x[1], reverse=False)
                selected_values = [i[0] for i in levenshtein_dist[:5] if i[1] < 0.5]
                if len(selected_values) == 0:
                    results[file] = "The table does not contain the keyterm {}.".format(keyterm)
                else:
                    results[file] = "The table does not contain the keyterm {}. The closest values are: {}".format(keyterm, selected_values)
    results = json.dumps(results, indent=4)
    results = """The requested information related to the keyterm "{}" is as follows:\n{}\nMaybe use the given information and write the code for next step.""".format(keyterm, results)
    return results

def _run_code_file(code_file: Path, timeout: Optional[int] = None) -> Tuple[int, str, str, float]:
    cmd = ["python", str(code_file)]
    env = os.environ.copy()
    cmd_str = " ".join(cmd)
    start_time = time.time()
    try:
        process = subprocess.Popen(
            cmd_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            shell=True,
            text=True
        )
        with tqdm(total=timeout, desc="Running code", unit="s") as pbar:
            while process.poll() is None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    process.kill()
                    execution_time = elapsed
                    return -1, "", f"Process timed out after {execution_time:.2f}s (timeout limit: {timeout}s)", execution_time
                pbar.update(1)
                time.sleep(1)
        stdout, stderr = process.communicate()
        execution_time = time.time() - start_time
        return process.returncode, stdout, stderr, execution_time
    except Exception as e:
        execution_time = time.time() - start_time
        return -1, "", str(e), execution_time

# validate the code
def validate_code(code: str) -> Dict[str, Any]:
    """
    Validate the generated code.

    Examples:
        validate_code("import numpy as np\na=[1,2,3]\nans=np.mean(a)")
    """
    success = True
    output = ""
    start_time = time.time()
    try:
        compile(code, '<string>', 'exec')
    except Exception as e:
        success = False
        error_msg = f"Compilation error occurred for the code: {str(e)}\nFull traceback:\n{traceback.format_exc()}"
        output = parse_and_truncate_error(error_msg)
        execution_time = time.time() - start_time
        logger.error("Compilation error during code validation.")
        return {
            "type": "code_execution",
            "status": "FAILED",
            "env_message": output,
            "execution_time": f"{execution_time:.2f}s"
        }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uuid_str = str(uuid.uuid4())
    code_file = os.path.join("./cache", f"validation_code_{timestamp}_{uuid_str}.py")
    # write the code to a file
    code_file = Path(code_file)
    if not 'print' in code and 'answer' in code:
        code += "\nprint(answer)"
    code_file.write_text(code)

    try:
        returncode, stdout, stderr, execution_time = _run_code_file(code_file, timeout=120)
        combined_output = stdout + ("\n" + stderr if stderr else "")
        if returncode != 0:
            success = False
            output = parse_and_truncate_error(combined_output)
        else:
            output = combined_output
    except Exception as e:
        success = False
        error_msg = f"Error executing code: {str(e)}\nFull traceback:\n{traceback.format_exc()}"
        output = parse_and_truncate_error(error_msg)
        execution_time = time.time() - start_time
        logger.error("Error executing code.")
    
    # delete code_file
    try:
        os.remove(code_file)
    except Exception as e:
        logger.error(f"Error deleting code file: {e}")

    logger.info(f"Code validation completed in {execution_time:.2f}s.")
    return {
        "type": "code_execution",
        "status": "SUCCESS" if success else "FAILED",
        "env_message": output,
        "execution_time": f"{execution_time:.2f}s"
    }

def debug(code, error_msg, debugger, history):
    costs = []
    summarization_prompt = """You are a bioinformatics expert. Your task is to summarize the following requested information:
<requested_information>
{requested_information}
</requested_information>
Please summarize the information in a concise and clear manner:
"""
    requested_info = []
    for obs in history:
        if obs['type'] == 'initial_observation':
            problem = obs["info"]["task_goal"]
        elif obs['type'] == "requested_info":
            requested_info.append(obs['env_message'])
    summarization_msg = summarization_prompt.format(requested_information='\n'.join(requested_info))
    summarization_msg = make_user_message(summarization_msg)
    summary, cost = debugger([summarization_msg])
    costs.append(cost)
    debug_prompt = """You are a Python debugging expert. Your task is to help the user debug their code.
The user is solving the problem: 
{problem}

The user has gotten the following knowledge:
{summary}

The user has provided the following code:
{code}

The user has encountered the following error:
{error_msg}

Please provide an explanation of the error and suggest a solution to fix it.
"""
    debug_msg = debug_prompt.format(
        problem=problem,
        summary=summary.content,
        code=code,
        error_msg=error_msg
    )
    debug_msg = make_user_message(debug_msg)
    response, cost = debugger([debug_msg])
    costs.append(cost)
    res = {"type": "debugging_info", "env_message": response.content+'\nPlease then use validate_code action to validate the debugged code.', "cost": costs}
    return res

def terminal(cmd: str):
    """
    Run a command in the terminal.

    Examples:
        terminal("ls -l")
    """
    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        execution_time = time.time() - start_time
        return {
            "type": "cmd_result",
            "status": "SUCCESS",
            "env_message": result.stdout,
            "execution_time": f"{execution_time:.2f}s"
        }
    except subprocess.CalledProcessError as e:
        return {
            "type": "cmd_result",
            "status": "FAILED",
            "env_message": f"Command '{cmd}' failed with error: {e.stderr}",
            "execution_time": f"{execution_time:.2f}s"
        }

if __name__ == "__main__":
    # Example usage
    code_file = "/work/10495/wenqishi/EHR-Gym/cache/demo_code.py"
    timeout = 60
    print(_run_code_file(code_file, timeout))