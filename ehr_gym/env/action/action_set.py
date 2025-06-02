import typing
from ehr_gym.env.action.base import AbstractActionSet
from ehr_gym.env.action.function import (
    request_info,
    validate_code,
    debug,
    terminal
)
from dataclasses import dataclass

@dataclass
class BasicAction:
    # entrypoint: callable
    signature: str
    description: str
    examples: list[str]

class BasicActionSet(AbstractActionSet):
    def __init__(self, custom_actions: typing.Optional[list[callable]] = None, strict: bool = False):
        super().__init__(strict)
        allowed_actions = [request_info, validate_code, debug]
        if custom_actions:
            allowed_actions.extend(custom_actions)
        allowed_actions = list(dict.fromkeys(allowed_actions).keys())

        # parse the action sand build the action space
        # self.action_set: dict[str, BasicAction] = {}
        self.action_set = {
            "request_info": request_info,
            "validate_code": validate_code,
            "debug": debug,
            "terminal": terminal,
        }
        # for func in allowed_actions:
        #     signature = f"{func.__name__}{inspect.signature(func)}"
        #     description, examples = action_docstring_parser.parse_string(func.__doc__)
    
    def example_action(self, action) -> str:
        """
        Returns an example action for the given action.
        """
        if action not in self.action_set:
            raise ValueError(f"Action {action} not found in action set.")
        return self.action_set[action].examples
    
    def describe(self, with_long_description: bool = True, with_examples: bool = True):
        """
        Returns a textual description of this action space.
        """
        description = f"""
{len(self.action_set)} different types of actions are available.

"""
        for _, action in self.action_set.items():
            description += f"""\
{action.signature}
"""
            if with_long_description:
                description += f"""\
    Description: {action.description}
"""
            if with_examples and action.examples:
                desriptions += f"""\
    Examples:
"""
                for example in action.examples:
                    description += f"""\
        {example}
"""
            example_action = self.example_action(action)
            if example_action:
                description += f"""\
    Example action: {example_action}
"""
        return description