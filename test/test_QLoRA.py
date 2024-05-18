# this program for profiling deepspeed train time

from datasets import load_dataset
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)
from accelerate import Accelerator
import torch
import time
from peft import LoraConfig

peft_config = LoraConfig(
    r=32,
    lora_alpha=8,
    bias="none",
    task_type="CAUSAL_LM",
)

# get dataset
dataset = load_dataset("imdb", split="train")

from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

device_map = {"": Accelerator().local_process_index}
print('device map: ', device_map)

model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-hf',
    quantization_config=bnb_config,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    use_flash_attention_2=True
    # trust_remote_code=script_args.trust_remote_code,
    # use_auth_token=script_args.use_auth_token,
)

# print("开始暂停")
# time.sleep(60)
# print("暂停结束")

training_args = TrainingArguments(
    output_dir='./output/test_QLoRA',
    gradient_checkpointing=False,
    bf16=True,
    per_device_train_batch_size=4,
    deepspeed='./config/ds.json',
)


# get trainer
trainer = SFTTrainer(
    model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    peft_config=peft_config,
)

# train
trainer.train()
