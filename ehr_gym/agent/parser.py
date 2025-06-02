import json
from typing import Tuple, Any

def parse_llm_response(response: str) -> Tuple[str, str]:
    """
    Parse the response from the LLM model.
    Args:
        response: The response from the LLM model.
    Returns:
        The response from the LLM model.
    """
    try:
        # First try to find JSON block in markdown code blocks
        if "```json" in response and "```" in response:
            # Extract content between ```json and ```
            start = response.find("```json") + 7
            end = response.find("```", start)
            if start > 6 and end > start:  # Valid positions found
                response = response[start:end].strip()
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1:
            # Extract the JSON part
            response = response[start:end + 1].strip()
        if '</think>' in response:
            # only keep the context after </think>
            response = response.split('</think>')[-1].strip()
        if '"""' in response:
            left = response.find('"""')
            right = response.rfind('"""')
            if left != -1 and right != -1 and left != right:
                # Extract the JSON part
                code = response[left + 3:right].strip()
            code = code.replace("\n", "\\n")
            response = response[:left].strip() + '\"' + code + '\"' + response[right + 3:].strip()
        # response = response.replace("\n", "\\n")
        # Validate JSON format
        if not response.strip():
            raise ValueError("Empty response")
            
        try:
            # response = response.encode('utf-8').decode('unicode_escape')
            response_dict = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}\nResponse: {response}")
            
        # Validate response structure
        if not isinstance(response_dict, dict):
            raise ValueError(f"Response must be a JSON object, got {type(response_dict)}")
            
        # Validate exact keys
        allowed_keys = {"action", "params"}
        actual_keys = set(response_dict.keys())
        if actual_keys != allowed_keys:
            raise ValueError(f"Response must contain exactly these keys: {allowed_keys}, got: {actual_keys}")
            
        action = response_dict["action"]
        params = response_dict["params"]
        
        # Validate action type
        valid_actions = {"request_info", "validate_code", "debug", "terminal"}
        if action not in valid_actions:
            raise ValueError(f"Invalid action '{action}'. Must be one of: {valid_actions}")
            
        # Validate params structure
        if not isinstance(params, dict):
            raise ValueError(f"'params' must be a dict, got {type(params)}")
            
        # Validate params content with strict key checking
        if action == "request_info":
            allowed_params_keys = {"data_path", "info_type", "keyterm"}
            actual_params_keys = set(params.keys())
            if actual_params_keys != allowed_params_keys:
                raise ValueError(f"'request_info' params must contain exactly: {allowed_params_keys}, got: {actual_params_keys}")
            if not isinstance(params["info_type"], str):
                raise ValueError(f"'info_type' must be string, got {type(params['info_type'])}")
        elif action == 'validate_code':  # validate_code or execute_code
            allowed_params_keys = {"code"}
            actual_params_keys = set(params.keys())
            if actual_params_keys != allowed_params_keys:
                raise ValueError(f"'{action}' params must contain exactly: {allowed_params_keys}, got: {actual_params_keys}")
            if not isinstance(params["code"], str):
                raise ValueError(f"'code' must be string, got {type(params['code'])}")
        elif action == "terminal":
            allowed_params_keys = {"cmd"}
            actual_params_keys = set(params.keys())
            if actual_params_keys != allowed_params_keys:
                raise ValueError(f"'{action}' params must contain exactly: {allowed_params_keys}, got: {actual_params_keys}")
            if not isinstance(params["cmd"], str):
                raise ValueError(f"'cmd' must be string, got {type(params['cmd'])}")
        else:
            allowed_params_keys = {"code", "error_msg"}
            actual_params_keys = set(params.keys())
            if actual_params_keys != allowed_params_keys:
                raise ValueError(f"'{action}' params must contain exactly: {allowed_params_keys}, got: {actual_params_keys}")
            if not isinstance(params["code"], str):
                raise ValueError(f"'code' must be string, got {type(params['code'])}")
        return action, params
        
    # except ValueError as e:
    #     return self._fix_parse_error(str(e), response)
    except Exception as e:
        raise ValueError(f"Failed to fix parse error: {e}")
        # return self._fix_parse_error(f"Unexpected error: {str(e)}", response)