# from transformer import

from numpy import False_
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from utils import ScriptArguments, format_prompt
import torch
from utils import DEFINE_EOS_TOKEN
import deepspeed

parser = HfArgumentParser(ScriptArguments)
train_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

model_name = train_args.model_name
instruction = train_args.prompt
max_new_tokens = train_args.max_new_tokens

device = 'cuda:0'
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # use_flash_attention_2=True,
    trust_remote_code=True,
    # load_in_4bit=True,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,)

tokenizer.eos_token = DEFINE_EOS_TOKEN
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token = DEFINE_EOS_TOKEN
model.config.pad_token_id = tokenizer.eos_token_id

input = format_prompt(instruction)
inputs = tokenizer(input, return_tensors='pt').to(device)
output = model.generate(inputs['input_ids'],max_new_tokens=max_new_tokens, do_sample=False)
output = tokenizer.decode(output[0], skip_special_tokens=True)

print(output)
