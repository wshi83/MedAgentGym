import yaml
import os
import json
import jsonlines

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def save_conversation_history(conversation_history, save_path):
    """
    Save the conversation history to a file.
    """
    with open(save_path, 'w') as f:
        json.dump(conversation_history, f, indent=4)