# from transformer import

from numpy import False_
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from utils import ScriptArguments, format_prompt
import torch
from utils import DEFINE_EOS_TOKEN, DEFINE_SEP_TOKEN, STEP_INSTRUCTION

parser = HfArgumentParser(ScriptArguments)
train_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

model_name = train_args.model_name
instruction = train_args.prompt
max_new_tokens = train_args.max_new_tokens
step_generate = train_args.step_generate

device = 'cuda:0'
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # use_flash_attention_2=True,
    trust_remote_code=True,
    # load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,)
model.generation_config.pad_token_id = tokenizer.pad_token_id

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
]


print(step_generate)
if step_generate:
    input = format_prompt(STEP_INSTRUCTION + instruction)
    inputs = tokenizer(input, return_tensors='pt')
    output = model.generate(inputs['input_ids'].to(device),
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=0.6,
                            top_p = 0.95,
                            eos_token_id=terminators)
    print(output)
    output = tokenizer.decode(output[0], skip_special_tokens=False) # set `skip_special_tokens=False` to debug
    # print(output)
    output = output.replace(DEFINE_SEP_TOKEN, " [SEP]\n")
    print(output)
else:
    input = format_prompt(instruction)
    inputs = tokenizer(input, return_tensors='pt')
    output = model.generate(inputs['input_ids'],
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            eos_token_id=terminators)
    output = tokenizer.decode(output[0], skip_special_tokens=False) # set `skip_special_tokens=False` to debug
    print(output)
