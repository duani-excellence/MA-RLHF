import os
import torch
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset
from accelerate import Accelerator
from peft import LoraConfig
import random
import re
random.seed(42)

os.environ["WANDB_PROJECT"] = "ma-rlhf"
os.environ["WANDB_RUN_NAME"] = "sft"

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)
from utils import (
    ScriptArguments,
    DEFINE_EOS_TOKEN,
    formatting_alpaca_func,
    is_main_process,
    create_peft,
)

parser = HfArgumentParser(ScriptArguments)
train_args: ScriptArguments = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]

dataset_name = train_args.dataset_name
model_name = train_args.model_name
deepspeed_config_name = train_args.deepspeed_config_name
seq_length = train_args.seq_length
batch_size = train_args.batch_size
output_name = train_args.output_name
is_peft = train_args.use_QLora
use_flash_attention_2 = train_args.use_flash_attention_2
dataset_sub_name = None
num_train_epochs = train_args.num_train_epochs
gradient_accumulation_steps = train_args.gradient_accumulation_steps
learning_rate = train_args.learning_rate

def create_datasets(dataset_name, dataset_sub_name):
    dataset = load_dataset(dataset_name)

    # merge two data
    is_merge_datasets = False
    if is_merge_datasets:
        dataset_hhrlhf = load_dataset('Anthropic/hh-rlhf', split='train')
        # dataset_hhrlhf = load_dataset('Anthropic/hh-rlhf', split='train[:50000]')
        def format_hhrlhf_to_alpaca(example):
            '''
            alpaca format : {instruction}, {input}, {output}
            hh rlhf format : {chosen}, {rejected}
                chosen: \n\nHuman: 11\n\nAssistant: 22 \n\nHuman: 33\n\nAssistant: 44
            target format is
                chosen: \n###Question: 11\n###Answer: 22 \n### Question 33\n###Answer: 44
                -> instruction: 【11\n###Answer: 22 \n### Question 33】
                -> output: 【 44】
            '''
            text = example["chosen"]
            text = re.sub(r'\n\nHuman:', '\n###Question:', text)
            text = re.sub(r'\n\nAssistant:', '\n###Answer:', text)
            text = text[1:]

            instruction = text.rsplit('\n###Answer:',1)[0]
            instruction = instruction.split('###Question:',1)[1]
            output = text.rsplit('\n###Answer:',1)[1]

            example['input'] = ''
            example['output'] = output
            example['instruction'] = instruction

            return example

        dataset_hhrlhf = dataset_hhrlhf.map(format_hhrlhf_to_alpaca, num_proc=8, remove_columns = ["chosen", "rejected"])
        dataset = concatenate_datasets([dataset['train'], dataset_hhrlhf])
        dataset = DatasetDict({'train': dataset})
        dataset = dataset.shuffle(seed=42)

    return dataset, None


def create_model_tokenizer(name):
    # QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    device_map = {"": Accelerator().local_process_index}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        use_flash_attention_2=use_flash_attention_2, # gpt 2 not support flash attention2
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                trust_remote_code=True,
                                                # padding_side='left',
                                                # model_max_length=1024
                                                )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.eos_token = DEFINE_EOS_TOKEN

    return model, tokenizer


def create_sft_datasets(datasets, tokenizer, format_func, seq_length=512):
    train_dataset = datasets["train"]

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_dataset,
        formatting_func=format_func,
        infinite=True,
        seq_length=seq_length,
        shuffle=True,
    )

    return train_dataset, None

def train():
    model, tokenizer = create_model_tokenizer(model_name)
    datasets, _ = create_datasets(dataset_name, dataset_sub_name)
    format_fun = formatting_alpaca_func
    train_datasets, _ = create_sft_datasets(datasets, tokenizer, format_fun, seq_length)

    # peft
    peft_config = create_peft(is_peft)

    training_args = TrainingArguments(
        output_dir=output_name,
        # save_strategy='steps',
        logging_steps=1,
        num_train_epochs=num_train_epochs,
        gradient_checkpointing=True,
        bf16=True,
        learning_rate=learning_rate,
        warmup_ratio=0.03,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        deepspeed=deepspeed_config_name,
        report_to='wandb',
        lr_scheduler_type='cosine',
        # max_steps=10,
    )

    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=train_datasets,
        max_seq_length=seq_length,
        peft_config=peft_config,
        packing=True,
        tokenizer=tokenizer,
        # formatting_func=format_fun,
    )
    trainer.train()
    trainer.save_model(output_name)


if __name__ == "__main__":
    train()
