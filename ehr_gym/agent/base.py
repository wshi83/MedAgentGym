from typing import Any, Dict, Union, List, Tuple
from dataclasses import dataclass
import json
import time
import tiktoken
from ehr_gym.llm.chat_api import OpenAIModelArgs, AzureModelArgs, VLLMModelArgs, make_system_message, make_user_message, make_assistant_message
from ehr_gym.agent.prompt import EHRPrompt, DynamicPrompt
from ehr_gym.agent.parser import parse_llm_response

@dataclass
class LLMConfig:
    """Configuration for the LLM agent."""
    model_path: str
    tokenizer_path: str
    max_length: int
    device: str
    port: int = 8000

class EHRAgent:
    """Agent class to interact with EHREnvironment using LLM"""
    def __init__(self, agent_config, permitted_actions):
        self.conversation_history = []
        self.agent_config = agent_config
        self.llm_config = agent_config['llm']

        if self.llm_config["model_type"] == 'OpenAI':
            self.llm = OpenAIModelArgs(
                model_name=self.llm_config["model_name"],
                max_total_tokens=self.llm_config["max_total_tokens"],
                max_input_tokens=self.llm_config["max_input_tokens"],
                max_new_tokens=self.llm_config["max_new_tokens"],
                vision_support=False,
            ).make_model()
        elif self.llm_config["model_type"] == 'Azure':
            self.llm = AzureModelArgs(
                model_name=self.llm_config["model_name"],
                temperature=self.llm_config["temperature"],
                max_new_tokens=self.llm_config["max_new_tokens"],
                deployment_name=self.llm_config["deployment_name"],
                log_probs=self.llm_config["log_probs"]
            ).make_model()
        elif self.llm_config["model_type"] == 'vLLM':
            self.llm = VLLMModelArgs(
                model_name=self.llm_config["model_name"],
                temperature=self.llm_config["temperature"],
                max_new_tokens=self.llm_config["max_new_tokens"],
                # deployment_name=self.llm_config["deployment_name"],
                port=self.llm_config["port"],
            ).make_model()
        else:
            raise ValueError("Model type {} not supported.".format(self.llm_config["model_type"]))
        self.conversation_history = []
        self.prompt = DynamicPrompt()
        self.parser = parse_llm_response
        self.cost = []
        self.permitted_actions = permitted_actions

    def act(self, obs: Any) -> Tuple[str, Dict[str, Any]]:
        """Main interface to get action from agent."""
        # try:
        if self.conversation_history == []:
            instruction = obs['info']['instruction']
            action_definitions = []
            action_formats = []
            for action in self.permitted_actions:
                action_definitions.append(self.prompt.action_definition[action])
                action_formats.append(self.prompt.action_format[action])
            action_definitions = "\n".join(action_definitions)
            action_formats = "\nor\n".join(action_formats)
            system_msg = self.prompt.prompt_template.format(instruction=instruction, action_definition=action_definitions, format_output=action_formats)
            user_msg = obs['info']['task_goal']
            if self.llm_config["model_type"] == 'vLLM':
                user_msg = make_user_message(content=system_msg+'\n'+user_msg)
            else:
                system_msg = make_system_message(content=system_msg)
                self.conversation_history.append(system_msg)
                user_msg = make_user_message(content=user_msg)
            self.conversation_history.append(user_msg)
        else:
            user_msg = obs['env_message']
            user_msg = make_user_message(content=user_msg)
            self.conversation_history.append(user_msg)
        # print(self.llm)
        # print(self.conversation_history)
        for i in range(self.agent_config['n_retry']):
            try:
                response, cost = self.llm(self.conversation_history)
                response = response.content
                self.cost.append(cost)
            except Exception as e:
                print('Error Message ', e)
                time.sleep(self.agent_config["retry_delay"])
                action, params = f"error: str({e})", {}   
                continue
            try:
                action, params = self.parser(response)
            except Exception as e:
                print('Error Message ', e)
                time.sleep(self.agent_config["retry_delay"])
                action, params = f"error: str({e})", {}  
                response = f"Error: {e}. Please regenerate the action."
            assistant_msg = make_assistant_message(content=response)
            self.conversation_history.append(assistant_msg)
            break
            # if e is ValueError, add a user message to conversation history
            # if not 'openai' in str(e).lower():
            #     self.conversation_history.append(make_user_message(content=f"Error: {e}. Please regenerate the action."))

        return action, params