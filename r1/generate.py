# from transformer import

from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
import torch

model_name = './output/qwen3_sft/checkpoint-407'

device = 'cuda:0'
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # use_flash_attention_2=True,
    trust_remote_code=True,
    # load_in_4bit=True,
    dtype=torch.bfloat16,
    device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,)


# input = format_prompt(instruction)
DEFINIED_SYSTEM_PROMPT = '你是小冬瓜智能体,请安全详细回答用户 USER 的问题'
input='简述强化学习PPO算法'
messages = [
            {'role':'system', 'content':DEFINIED_SYSTEM_PROMPT},
            {'role':'user', 'content': input},
        ]


prompt = tokenizer.apply_chat_template([messages], 
                                       tokenize=False, 
                                       add_generation_prompt=True)
print(prompt)
inputs = tokenizer(prompt,return_tensors='pt')
output = model.generate(inputs['input_ids'].to(device),
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=1.0,
                        )
output = tokenizer.decode(output[0], skip_special_tokens=False) # set `skip_special_tokens=False` to debug
print(output)
