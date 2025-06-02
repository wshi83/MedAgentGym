import logging
from typing import Literal
import time
logger = logging.getLogger(__name__)

class Chat:
    def __init__(self):
        self.messages = []
    
    def add_message(
        self, role: Literal["user", "assistant", "info"], content: str
    ):
        """
        Add a message to the chatbox
        """
        utc_time = time.time()
        if role not in ("user", "assistant", "info"):
            raise ValueError("Role must be 'user', 'assistant', or 'info'")
        else:
            self.messages.append({"role": role, "timestamp": utc_time, "message": content})
        timestamp = time.strftime("%H:%M:%S", time.localtime(utc_time))