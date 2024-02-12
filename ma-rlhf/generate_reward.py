# from transformer import

# from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from utils import ScriptArguments, format_prompt
import torch
from utils import DEFINE_EOS_TOKEN

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoModelForSequenceClassification,
)

parser = HfArgumentParser(ScriptArguments)
train_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

model_name = train_args.model_name

device = 'cuda:0'
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, torch_dtype=torch.float16, use_flash_attention_2=False, trust_remote_code=True, num_labels=1
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.eos_token = DEFINE_EOS_TOKEN
model.config.eos_token = DEFINE_EOS_TOKEN
model.config.eos_token_id = tokenizer.eos_token_id


input = ''

inputs = tokenizer(input, return_tensors='pt').to(device)
print('[format prompt]:', inputs)
output = model.generate(inputs['input_ids'], max_new_tokens=max_new_tokens)
output = tokenizer.decode(output[0], skip_special_tokens=True)

print(output)
