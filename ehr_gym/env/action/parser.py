import ast
import pyparsing as pp
from dataclasses import dataclass
from typing import Any

# parse the docstring in actions, in order to get the action's name and its parameters
action_docstring_parser = pp.ParserElement(
    pp.Group(pp.OneOrMore(pp.Word(pp.printables), stop_on=pp.Literal("Examples:")))
    + pp.Literal("Examples:").suppress()
)