import os
import re
import json
import time
import subprocess
from dataclasses import dataclass, field
from abc import ABC
import pandas as pd
import numpy as np
import datetime as datetime
import gymnasium as gym
from gymnasium import spaces
import logging
from ehr_gym.env.task.base import AbstractEHRTask
from ehr_gym.env.spaces import AnyDict, Float, Unicode
from ehr_gym.env.chat import Chat
from ehr_gym.env.action.action_set import BasicActionSet
from ehr_gym.llm.chat_api import AzureModelArgs
from typing import Any, Optional, Callable


logging.basicConfig(
    level=logging.INFO, format="%(name)s : %(levelname)-8s : %(message)s"
)
logger = logging.getLogger(__name__)

# Base record classes for different types of environment interactions
@dataclass
class BaseRecord:
    timestamp: datetime
    code: str = None
    output: str = None
    success: bool = None
    execution_time: float = None

@dataclass
class InfoRequestRecord(BaseRecord):
    info_type: str = field(default="")
    content: Any = None

class EHREnv(gym.Env, ABC):
    """The main EHRGym class, which encapsulates instruction-following EHR question-answering into a Gymnasium environment."""

    # gym metdata
    metadata = {"render_modes": None}

    def __init__(
            self,
            # task-related arguments
            task_entrypoint: type[AbstractEHRTask],
            task_kwargs: dict = {
                "data_path": None,
                "debugger_config": None,
            },
            # agent-related arguments
            action_mapping: Optional[Callable] = BasicActionSet().action_set,
        ):
        """
        Instantiate a ready to use EHREnv gym environment.

        Args:
            task_entrypoint: a callable that returns a new task object from a task_id
            task_kwargs: additional arguments passed `task-entrypoint`.
        """
        super().__init__()
        self.task_entrypoint = task_entrypoint
        self.task_kwargs = dict(**task_kwargs)
        self.task = None

        # observation space
        self.observation_space = gym.spaces.Dict(
            {
                "chat_messages": gym.spaces.Sequence(
                    gym.spaces.Dict(
                        {
                            "role": Unicode(),
                            "timestamp": Float(),
                            "message": Unicode(),
                        }
                    )
                ),
                "goal": Unicode(),
                "history": AnyDict(),
                "elapsed_time": Float(),
            }
        )

        # action space
        self.action_space = Unicode()
        self.action_mapping = action_mapping

        # initialize chat
        self.chat: Chat = None

        self.terminate_on_infeasible = False

        debugger_config = self.task_kwargs.get("debugger_config", None)
        self.debugger = AzureModelArgs(
            model_name=debugger_config["model_name"],
            temperature=debugger_config["temperature"],
            max_new_tokens=debugger_config["max_new_tokens"],
            deployment_name=debugger_config["deployment_name"],
            log_probs=debugger_config["log_probs"],
        ).make_model()
        self.env_history = []
        self.need_context = False

    def close(self):
        if self.task:
            self.task.teardown()
            self.task = None
        if self.chat:
            self.chat.close()
            self.chat = None
        if self.env_history != []:
            self.env_history = []
    
    def reset(self, task_id, *args, **kwargs):
        """
        Reset the environment to a new task.
        """

        super().reset(*args, **kwargs)
        if self.task:
            self.task.teardown()
            self.task = None
        if self.chat:
            self.chat.close()
            self.chat = None
        self.env_history = []
        
        # create a new task
        self.task = self.task_entrypoint(task_id=task_id, **self.task_kwargs)

        def override_property(task, env, property):
            """Extract property value from env if not None, otherwise from task."""
            env_value = getattr(env, property)
            task_value = getattr(task, property)
            if env_value is None:
                return task_value
            else:
                if task_value is not None:
                    logger.warning(
                        f"Overriding the task's {property} parameter ({repr(task_value)} => {repr(env_value)}). This might change the task's behaviour and difficulty."
                    )
                return env_value
        
        # fetch task's desired parameters for setup
        self.chat = Chat()
        self.chat.add_message(
            role="assistant",
            content="Hi! I am your EHR assistant, I can perform tasks based on the EHR data. What can I help you with?"
        )

        # setup the task goal
        task_goal, task_info = self.task.setup()

        # no goal specified
        if task_goal is None:
            self.goal_object = []
        # convert text=only goal (legacy) to new format
        elif isinstance(task_goal, str):
            self.goal_object = [{"type": "text", "text": task_goal}]
        # new format goal with multiple texts and images (OpenAI style)
        elif isinstance(task_goal, list):
            self.goal_object = task_goal
        else:
            raise ValueError(f"task_goal should be of type str or list, got {task_goal.__class__}")

        # send the task goal (if any) to the chat
        for message in self.goal_object:
            match message["type"]:
                case "text":
                    self.chat.add_message(role="user", content=message["text"])
                case _:
                    raise ValueError(f"Unsupported message type: {message['type']}")

        
        # init start time
        self.start_time = time.time()

        # no action yet
        self.history = []
        self.action = None

        # extract obs and information from the environment
        obs = self._get_obs()
        if "context_info" in obs["info"]:
            self.need_context = True
        else:
            self.need_context = False
        self.env_history.append(obs)
        info = {}
        info["task_goal"] = task_goal
        info["task_info"] = task_info

        return obs, info
    
    def step(self, action: str, **kwargs) -> tuple:
        """
        Take a step in the environment.

        Args:
            action: the action to take in the environment.

        Returns:
            obs: the observation of the environment.
            reward: the reward of the environment.
            done: whether the environment is done.
            info: additional information about the environment.
        """
        # save the action

        self.last_action = action
        if 'error' in action:
            obs = {}
            obs["type"] = "compile_error"
            obs['env_message'] = action
            self.env_history.append(obs)
            return obs, 0, False, False, {"message": action}

        info = {}
        info["action_exec_start"] = time.time()
        info["action_exec_timeout"] = 0

        def send_message_to_user(text: str):
            if not isinstance(text, str):
                raise ValueError(f"Forbidden value: {text} is not a string")
            self.chat.add_message(role="assistant", content=text)
        
        def report_infeasible_instructions(reason: str):
            if not isinstance(reason, str):
                raise ValueError(f"Forbidden value: {reason} is not a string")
            self.chat.add_message(role="infeasible", content=reason)
            self.infeasible_message_received = True
        
        # try to execute the action
        logger.debug(f"Executing action")
        try:
            action_function = self.action_mapping[action]
            if action == 'debug':
                # add one argument to kwargs
                kwargs["debugger"] = self.debugger
                kwargs["history"] = self.env_history
            elif action == 'validate_code' and self.need_context:
                kwargs["code"] = self.task.context.replace(self.task.context_pattern, '\n'+kwargs['code']+'\n')
                # print(kwargs["code"])
            results = action_function(**kwargs)
        except Exception as e:
            # print(f"Error: {e}")
            self.last_action_error = f"{type(e).__name__}: {e}"
            match = re.match("TimeoutError: timeout of ([0-9.]+)ms", self.last_action_error)
            if match:
                info["action_exec_timeout"] = float(match.groups()[0]) / 1000
            results = {
                'type': 'error',
                'env_message': f"Error: {self.last_action_error}",
            }
        logger.debug(f"Action executed")
        info["action_exec_stop"] = time.time()

        # extract observation (generic)
        obs = self._get_obs(results)
        self.env_history.append(obs)
        logger.debug(f"Observation extracted")

        # extract reward, done, user_message, info (task_specific)
        logger.debug(f"Initiating task validation")
        reward, done, user_message, task_info = self._task_validate(obs)
        info["task_info"] = task_info
        logger.debug(f"Task validated")

        # new step API wants a 5-tuple (gymnasium style)
        terminated = done or (
            self.terminate_on_infeasible and self.infeasible_message_received
        ) # task or agent can terminate the episode
        truncated = False
        return obs, reward, terminated, truncated, info
    
    def _task_validate(self, obs):
        reward, done, user_message, task_info = self.task.validate(self.chat.messages, obs)
        return reward, done, user_message, task_info
    
    def _get_obs(self, results=None):
        if results is not None:
            if type(results) == str:
                obs = {}
                obs['type'] = "requested_info"
                obs['env_message'] = results
            elif type(results) == dict:
                if 'type' in results and 'env_message' in results:
                    obs = results
                else:
                    if not 'env_message' in results:
                        obs['env_message'] = json.dumps(results, indent=4)
                    else:
                        obs['env_message'] = results['env_message']
                    if not 'type' in results:
                        obs['type'] = "environment_feedback"
                    else:
                        obs['type'] = results['type']
            else:
                raise ValueError(f"Unsupported type: {type(results)}")
            return obs
        else:
            return self.task._get_obs()