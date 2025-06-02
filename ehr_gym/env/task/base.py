from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

class AbstractEHRTask(ABC):
    """
    Abstract class for EHR tasks
    """

    @classmethod
    def get_task_id(cls):
        raise NotImplementedError
    
    def __init__(self, task_id: int) -> None:
        self.task_id = task_id
        
    @abstractmethod
    def setup(self):
        """
        Set up everything needed to exectue the task.

        Args:
            data_path: path to the data
        
        Returns:
            goal: str, goal of the task
            info: dict, custom information from the task
        """
    
    def teardown(self) -> None:
        """
        Tear down the task and clean up any resource / data created by the task (optional).
        """
        pass

    
    def setup_goal(self) -> tuple[str, dict]:
        return "", {}