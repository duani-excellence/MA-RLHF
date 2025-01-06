from utils import (
    ScriptArguments,
    DEFINE_PAD_TOKEN,
    is_main_process,
    create_peft,
    DEFINE_SEP_TOKEN,
)

from data_prm import (
    process_sft_step,
    DataCollatorForSFT
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    Trainer
)

from peft import (
    LoraModel,
    LoraConfig,
)

import os
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from accelerate import Accelerator
import random
random.seed(42)

os.environ["WANDB_PROJECT"] = "ma-rlhf"
os.environ["WANDB_RUN_NAME"] = "sft-step"


parser = HfArgumentParser(ScriptArguments)
train_args: ScriptArguments = parser.parse_args_into_dataclasses(
    return_remaining_strings=True)[0]

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


def create_sft_step_datasets(dataset_name, tokenizer, seq_length=1024):
    dataset = load_dataset(dataset_name)
    print(dataset)
    dataset = dataset['train'].map(process_sft_step,
                           num_proc=24,
                           remove_columns=[
                               'prompt', 'completions', 'labels'],
                            fn_kwargs={ "tokenizer": tokenizer},
                            batched = False,
                           )

    dataset = dataset.filter(lambda x: len(x["input_ids"]) < seq_length, batched=False)
    print(dataset)
    return dataset, None


def create_model_tokenizer(model_name):
    # QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    device_map = {"": Accelerator().local_process_index}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config if is_peft else None,
        device_map=device_map,
        use_flash_attention_2=use_flash_attention_2,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              add_bos_token=False,
                                              padding=True,
                                              truncation=True)
    tokenizer.add_special_tokens({'pad_token': DEFINE_PAD_TOKEN})
    tokenizer.add_special_tokens({'sep_token': DEFINE_SEP_TOKEN})
    model.pad_token_id = tokenizer.pad_token_id
    model.pad_token = tokenizer.pad_token

    return model, tokenizer



def train():
    model, tokenizer = create_model_tokenizer(model_name)
    train_datasets, _ = create_sft_step_datasets(dataset_name, tokenizer, seq_length=seq_length)
    collator = DataCollatorForSFT(tokenizer=tokenizer)

    # debug data
    # data_loader = DataLoader(
    #     train_datasets,                   # 数据集
    #     batch_size=2,                     # 批次大小
    #     shuffle=True,                     # 是否打乱数据
    #     collate_fn=collator,              # 自定义数据整理函数
    # )
    # for batch in data_loader:
    #     print(batch)
    #     break
    # return

    # peft
    peft_config = create_peft(is_peft)
    model.add_adapter(peft_config)

    training_args = TrainingArguments(
        output_dir=output_name,
        save_strategy='epoch',
        logging_steps=1,
        num_train_epochs=num_train_epochs,
        gradient_checkpointing=True,
        bf16=True,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        deepspeed=deepspeed_config_name,
        report_to='wandb',
        lr_scheduler_type='cosine',
        # max_steps=100,
    )

    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=train_datasets,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(output_name)
    tokenizer.save_pretrained(output_name)


if __name__ == "__main__":
    train()
