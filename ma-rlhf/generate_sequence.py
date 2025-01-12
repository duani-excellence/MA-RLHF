# from transformer import

from numpy import False_, negative, positive
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from utils import ScriptArguments, format_prompt
import torch
from utils import DEFINE_EOS_TOKEN, DEFINE_SEP_TOKEN, DEFINE_POSITIVE_TOKEN, DEFINE_NEGATIVE_TOKEN

parser = HfArgumentParser(ScriptArguments)
train_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

model_name = train_args.model_name
instruction = train_args.prompt
max_new_tokens = train_args.max_new_tokens
step_generate = train_args.step_generate

# device = 'cuda:0'
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # use_flash_attention_2=True,
    trust_remote_code=True,
    # load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,)

# tokenizer.eos_token = DEFINE_EOS_TOKEN
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token_id = tokenizer.eos_token_id
# model.config.pad_token = DEFINE_EOS_TOKEN
# model.config.pad_token_id = tokenizer.eos_token_id
# model.pad_token_id = tokenizer.pad_token_id
# model.pad_token = tokenizer.pad_token

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
]

input = format_prompt(instruction)
inputs = tokenizer(input, return_tensors='pt')

output = model(inputs['input_ids'], inputs['attention_mask'])
logits = output.logits # shape: (batch_size, seq_len, vocab_size)
idx = torch.where(inputs['input_ids'] == tokenizer.convert_tokens_to_ids(DEFINE_SEP_TOKEN))[0]

positive_token_id = tokenizer.convert_tokens_to_ids(DEFINE_POSITIVE_TOKEN)
negative_token_id = tokenizer.convert_tokens_to_ids(DEFINE_NEGATIVE_TOKEN)

logits_verifier = logits[0, idx, [positive_token_id, negative_token_id]]

sep_prob = torch.nn.functional.softmax(logits_verifier, dim=1)
print(sep_prob)
_, pred = torch.max(sep_prob, dim=1)
pred = pred.tolist()
print(pred)
result = [ bool(p) for p in pred ]
print(result)
