# For Mac : DEEPSPEED_BACKEND=gloo deepspeed train.py

from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFINIED_SYSTEM_PROMPT = '你是小冬瓜智能体,请安全详细回答用户 USER 的问题'

model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')
tokenizer =  AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')

dataset = load_dataset('tatsu-lab/alpaca')
def map_cat_inst_input(example):
    example['messages'] = [
        {'role':'system', 'content':DEFINIED_SYSTEM_PROMPT},
        {'role':'user', 'content': example['instruction']+example['input']},
        {'role':'assistant', 'content': example['output']},
    ]
    return example

dataset_alpaca = dataset.map(map_cat_inst_input,
                             remove_columns=["instruction", "input", "output", "text"])

config = SFTConfig(
    output_dir="output/qwen3_sft",
    per_device_train_batch_size = 2,
    max_length = 256,
    max_steps = 10,
    bf16=False,
    fp16=True,
    # deepspeed='./ds.json',
)

trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=dataset['train'],
)
trainer.train()