import base64
import collections
import io
import json
import logging
import os
import re
import time
from copy import deepcopy
from functools import cache
from typing import TYPE_CHECKING, Any, Union
from warnings import warn

import numpy as np
import tiktoken
import yaml
from langchain.schema import BaseMessage
from langchain_community.adapters.openai import convert_message_to_dict
from PIL import Image
from transformers import AutoModel, AutoTokenizer

class BaseMessage(dict):
    def __init__(self, role: str, content: Union[str, list[dict]], **kwargs):
        allowed_attrs = {"log_probs"}
        invalid_attrs = set(kwargs.keys()) - allowed_attrs
        if invalid_attrs:
            raise ValueError(f"Invalid attributes: {invalid_attrs}")
        self["role"] = role
        self["content"] = deepcopy(content)
        self.update(kwargs)

    def __str__(self, warn_if_image=False) -> str:
        if isinstance(self["content"], str):
            return self["content"]
        if not all(elem["type"] == "text" for elem in self["content"]):
            msg = "The content of the message has images, which are not displayed in the string representation."
            if warn_if_image:
                logging.warning(msg)
            else:
                logging.info(msg)

        return "\n".join([elem["text"] for elem in self["content"] if elem["type"] == "text"])

    def add_content(self, type: str, content: Any):
        if isinstance(self["content"], str):
            text = self["content"]
            self["content"] = []
            self["content"].append({"type": "text", "text": text})
        self["content"].append({"type": type, type: content})

    def add_text(self, text: str):
        self.add_content("text", text)

    def add_image(self, image: np.ndarray | Image.Image | str, detail: str = None):
        if not isinstance(image, str):
            image_url = image_to_jpg_base64_url(image)
        else:
            image_url = image
        if detail:
            self.add_content("image_url", {"url": image_url, "detail": detail})
        else:
            self.add_content("image_url", {"url": image_url})

    def to_markdown(self):
        if isinstance(self["content"], str):
            return f"\n```\n{self['content']}\n```\n"
        res = []
        for elem in self["content"]:
            # add texts between ticks and images
            if elem["type"] == "text":
                res.append(f"\n```\n{elem['text']}\n```\n")
            elif elem["type"] == "image_url":
                img_str = (
                    elem["image_url"]
                    if isinstance(elem["image_url"], str)
                    else elem["image_url"]["url"]
                )
                res.append(f"![image]({img_str})")
        return "\n".join(res)

    def merge(self):
        """Merges content elements of type 'text' if they are adjacent."""
        if isinstance(self["content"], str):
            return
        new_content = []
        for elem in self["content"]:
            if elem["type"] == "text":
                if new_content and new_content[-1]["type"] == "text":
                    new_content[-1]["text"] += "\n" + elem["text"]
                else:
                    new_content.append(elem)
            else:
                new_content.append(elem)
        self["content"] = new_content
        if len(self["content"]) == 1:
            self["content"] = self["content"][0]["text"]

class AIMessage(BaseMessage):
    def __init__(self, content: Union[str, list[dict]], log_probs=None):
        super().__init__("assistant", content, log_probs=log_probs)