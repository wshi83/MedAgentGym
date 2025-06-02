import logging
import os
import re
import time
from dataclasses import dataclass
from functools import partial
from typing import Optional
import openai
from openai import AzureOpenAI, OpenAI
from abc import ABC, abstractmethod
from .message import AIMessage

def make_system_message(content: str) -> dict:
    return dict(role="system", content=content)

def make_user_message(content: str) -> dict:
    return dict(role="user", content=content)

def make_assistant_message(content: str) -> dict:
    return dict(role="assistant", content=content)

class AbstractChatModel(ABC):
    @abstractmethod
    def __call__(self, messages: list[dict]) -> dict:
        pass
    
    def get_stats(self):
        return {}

@dataclass
class BaseModelArgs(ABC):
    """Base class for all model arguments"""
    model_name: str
    max_total_tokens: int = None
    max_input_tokens: int = None
    max_new_tokens: int = None
    temperature: float = 0.6
    vision_support: bool = False
    log_probs: bool = False

    @abstractmethod
    def make_model(self) -> AbstractChatModel:
        pass
    
    def prepare_server(self):
        pass
    
    def close_server(self):
        pass

@dataclass
class OpenAIModelArgs(BaseModelArgs):
    """Serializable object or instantiating a generic chat model with an OpenAI model"""
    def make_model(self):
        return OpenAIChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            log_probs=self.log_probs
        )

@dataclass
class AzureModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an Azure model."""
    deployment_name: str = None
    def make_model(self):
        return AzureChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            deployment_name=self.deployment_name,
            log_probs=self.log_probs
        )

@dataclass
class VLLMModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an local-host vLLM model."""
    port: int = 8000
    def make_model(self):
        return VLLMChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            # deployment_name=self.deployment_name,
            port=self.port
        )

class ChatModel(AbstractChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
        max_retry=4,
        min_retry_wait_time=60,
        api_key_env_var=None,
        client_class=OpenAI,
        client_args=None,
        pricing_func=None,
        log_probs=False,
    ):
        assert max_retry > 0, "max_retry should be greater than 0"

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retry = max_retry
        self.min_retry_wait_time = min_retry_wait_time
        self.log_probs = log_probs

        # Get the API key from the environment variable if not provided
        if api_key_env_var:
            api_key = api_key or os.getenv(api_key_env_var)
        self.api_key = api_key

        client_args = client_args or {}
        self.client = client_class(
            api_key=api_key,
            **client_args,
        )
    
    def __call__(self, messages: list[dict], n_samples: int = 1, temperature: float = None) -> dict:
        # Initialize retry tracking attributes
        self.retries = 0
        self.success = False
        self.error_types = []

        completion = None
        e = None
        for itr in range(self.max_retry):
            self.retries += 1
            temperature = temperature if temperature is not None else self.temperature
            try:
                if "o3" in self.model_name or "o4" in self.model_name:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        n=n_samples,
                        logprobs=self.log_probs
                    )
                elif "qwen3" in self.model_name.lower():
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        n=n_samples,
                        temperature=temperature,
                        top_p=0.95,
                        max_tokens=self.max_tokens,
                        logprobs=self.log_probs
                    )
                else:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        n=n_samples,
                        temperature=temperature,
                        max_tokens=self.max_tokens,
                        logprobs=self.log_probs
                    )
                self.success = True
                break
            except openai.OpenAIError as e:
                print(e)
                error_type = handle_error(e, itr, self.min_retry_wait_time, self.max_retry)
                self.error_types.append(error_type)
        if not completion:
            raise RetryError(
                f"Failed to get a response from the API after {self.max_retry} retries\n"
                f"Last error: {error_type}"
            )
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        cost = {'input_tokens': input_tokens, 'completion_tokens': output_tokens}
        if n_samples == 1:
            res = completion.choices[0].message
            return res, cost
            # res = AIMessage(completion.choices[0].message.content)
            # if self.log_probs:
            #     res["log_probs"] = completio.choices[0].log_probs
            # return res
        else:
            return [c.message for c in completion.choices], cost
    
    def get_stats(self):
        return {
            "n_retry_llm": self.retries,
        }

class OpenAIChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
        max_retry=4,
        min_retry_wait_time=60,
        log_probs=False
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=max_retry,
            min_retry_wait_time=min_retry_wait_time,
            api_key_env_var="OPENAI_API_KEY",
            client_class=OpenAI,
            log_probs=log_probs
        )

class AzureChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        deployment_name=None,
        temperature=0.5,
        max_tokens=100,
        max_retry=4,
        min_retry_wait_time=60,
        log_probs=False,
    ):
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("API_VERSION")
        assert endpoint, "AZURE_OPENAI_ENDPOINT has to be defined in the environment"

        client_args = {
            "azure_deployment": deployment_name,
            "azure_endpoint": endpoint,
            "api_version": api_version,
        }
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=max_retry,
            min_retry_wait_time=min_retry_wait_time,
            client_class=AzureOpenAI,
            client_args=client_args,
            log_probs=log_probs,
        )

class VLLMChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
        n_retry_server=4,
        min_retry_wait_time=60,
        port=8000,
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=n_retry_server,
            min_retry_wait_time=min_retry_wait_time,
            api_key_env_var="VLLM_API_KEY",
            client_class=OpenAI,
            client_args={"base_url": "http://0.0.0.0:{}/v1".format(port)},
            pricing_func=None,
        )