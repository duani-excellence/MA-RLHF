# from transformer import

from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from utils import ScriptArguments, format_prompt
import torch
from utils import DEFINE_EOS_TOKEN

parser = HfArgumentParser(ScriptArguments)
train_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

model_name = train_args.model_name
instruction = train_args.prompt
max_new_tokens = train_args.max_new_tokens

device = 'cuda:0'
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, use_flash_attention_2=False, trust_remote_code=True,
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.eos_token = DEFINE_EOS_TOKEN
model.config.eos_token = DEFINE_EOS_TOKEN
model.config.eos_token_id = tokenizer.eos_token_id

# instruction = '我心情不好，有点头痛该怎么办？'
# input = f'###Question: {instruction}\n ###Answer: '
input = format_prompt(instruction)

inputs = tokenizer(input, return_tensors='pt').to(device)
print('[format prompt]:', inputs)
output = model.generate(inputs['input_ids'], max_new_tokens=max_new_tokens)
output = tokenizer.decode(output[0], skip_special_tokens=True)

print(output)

'''

###Question: 我心情不好，有点头痛该怎么办？
 ###Answer:
心情不好的时候，有头痛、头晕、头痛、头痛和头痛，这些症状都是正常的，我们需要继续努力抗拒抑郁和躁动，努力克服痛苦。
心情不好的时候，可以采取以下方法：
1. 尽量抑制自己的情绪，不要过度感受自己的情绪，不要过度感受自己的情绪，不要过度感受自己的情绪，不要过度感受自己的情绪，不要过度感受自己的情绪，不要过度感受自己的情绪，不要过度感受自己的情绪。
2. 尽量让自己的身体和精神更健康，尽量减少不必要的压力，尽量减少不必要的压力，尽量减少不必要的压力，尽量减少不必要的压力，尽量减少不必要的压力，尽量减少不必要的压力。
3. 尽量减少喝酒，尽量减少喝酒，尽量减少喝酒，尽量减少喝酒，尽量减少喝酒，尽量减少喝酒。
4. 尽量减少吃

'''
