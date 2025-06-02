from abc import ABC, abstractmethod
class AbstractActionSet(ABC):
    def __init__(self, strict: bool = False):
        self.strict = strict

    @abstractmethod
    def describe(self, with_long_description: bool = True, with_examples: bool = True) -> str:
        """
        Returns a textual description of the action space.
        """
    
    @abstractmethod
    def example_action(self, action) -> str:
        """
        Returns an example action as a string.
        """
    
    # @abstractmethod
    # def to_python_code(self, action) -> str:
    #     """
    #     Converst the given action to python code.

    #     Args:
    #         action: the action to convert.
        
    #     Returns:
    #         Executable python code that performs the action in browsergym environment.
    #     """

def execute_python_code(
    code: str,
    send_message_to_user: callable,
    report_infeasible_instuctions: callable,
):
    """
    Executes Python code in a new context

    Args:
        code: the python code to execute, as a string.
        send_message_to_user: a function that sends a message to the user.
        report_infeasible_instuctions: a function that reports infeasible instructions.
    """
    globals = {
        "send_message_to_user": send_message_to_user,
        "report_infeasible_instuctions": report_infeasible_instuctions,
    }
    exec(code, globals)