import sys

# sys.path.append('./../src/')
from utils import ScriptArguments
from transformers import HfArgumentParser

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses(
    return_remaining_strings=True)[0]

print(script_args.reward_model_name)
