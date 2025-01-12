# from transformer import

from numpy import False_, negative, positive
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from peft import PeftModel
from utils import ScriptArguments, format_prompt
import torch
from utils import DEFINE_EOS_TOKEN, DEFINE_SEP_TOKEN, DEFINE_POSITIVE_TOKEN, DEFINE_NEGATIVE_TOKEN, STEP_INSTRUCTION, PRM_INSTRUCTION
# import deepspeed

parser = HfArgumentParser(ScriptArguments)
train_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

model_name = train_args.model_name
prompt = train_args.prompt
max_new_tokens = train_args.max_new_tokens
step_generate = train_args.step_generate
lora_path = train_args.lora_path

device = 'cuda:0'
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # use_flash_attention_2=True,
    trust_remote_code=True,
    # load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,)
model.generation_config.pad_token_id = tokenizer.pad_token_id

input = format_prompt(STEP_INSTRUCTION + prompt)
inputs = tokenizer(input, return_tensors='pt')


terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
]

print('-'*100)
print('stage step generate')
print('-'*100)
prompt_len = inputs['input_ids'].shape[1]
model.to(device)
# if step_generate:
with torch.no_grad():
    output = model.generate(input_ids = inputs['input_ids'].to(device),
                            attention_mask = inputs['attention_mask'].to(device),
                            max_new_tokens = max_new_tokens,
                            do_sample = True,
                            temperature = 0.6,
                            top_p = 0.95,
                            eos_token_id = terminators)


generative_string = tokenizer.decode(output[0], skip_special_tokens=False) # set `skip_special_tokens=False` to debug
generative_string = generative_string.replace(DEFINE_SEP_TOKEN, " [SEP]\n")
print(generative_string)

response = output[:, prompt_len:]


# prm score

print('-'*100)
print('stage prm score')
print('-'*100)
input = format_prompt(PRM_INSTRUCTION + prompt)
inputs = tokenizer(input, return_tensors='pt')
inputs['input_ids'] = torch.cat([inputs['input_ids'].to(device), response.to(device)], dim=1)


model.to(device)
model = PeftModel.from_pretrained(model, lora_path)
with torch.no_grad():
    output = model(inputs['input_ids'].to(device))
logits = output.logits # shape: (batch_size, seq_len, vocab_size)
idx = torch.where(inputs['input_ids'][0,:] == tokenizer.convert_tokens_to_ids(DEFINE_SEP_TOKEN))[0]

positive_token_id = tokenizer.convert_tokens_to_ids(DEFINE_POSITIVE_TOKEN)
negative_token_id = tokenizer.convert_tokens_to_ids(DEFINE_NEGATIVE_TOKEN)
logits_idx = logits[0, idx]
logits_idx_token = logits_idx[:, [positive_token_id, negative_token_id]] # positive token 对应 False, negative token 对应 True


sep_prob = torch.nn.functional.softmax(logits_idx_token, dim=1)
probs, preds = torch.max(sep_prob, dim=1)
preds = preds.tolist()
probs = probs.tolist()
result = [ bool(p) for p in preds ]

response_string = tokenizer.decode(response[0], skip_special_tokens=False)
step_string = response_string.split(DEFINE_SEP_TOKEN)

i = 0
for step, label, prob in zip(step_string, result, probs):
    print(f"Step: No.{i}, Label: {label}, Prob: {prob}")
    print(f"Step: {step}")
    i = i + 1
    print('-'*20)
