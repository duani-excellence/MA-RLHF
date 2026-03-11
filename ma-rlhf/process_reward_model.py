from utils import (
    ScriptArguments,
    DEFINE_PAD_TOKEN,
    is_main_process,
    create_peft,
    DEFINE_SEP_TOKEN,
    create_peft_lm_head,
    create_peft_prm_lm_head,
)

from data_prm import (
    process_prm_step,
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
    get_peft_model,
)

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from accelerate import Accelerator
import random
random.seed(42)

os.environ["WANDB_PROJECT"] = "ma-rlhf"
os.environ["WANDB_RUN_NAME"] = "prm-step"


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


def create_prm_step_datasets(dataset_name, tokenizer, seq_length=1024):
    dataset = load_dataset(dataset_name)
    # dataset = dataset.filter(lambda x: x['type'] == 'PRM800K' and  x['labels'][-1] == False, batched=False)
    print(dataset)
    dataset = dataset['train'].map(process_prm_step,
                           num_proc=24,
                           remove_columns=[
                               'prompt', 'completions', 'labels', 'is_step', 'is_end', 'type'],
                            fn_kwargs={ "tokenizer": tokenizer},
                            batched = False,
                            load_from_cache_file=False,
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
                                              add_bos_token=True,
                                              padding=True,
                                              truncation=True,
                                              use_cache=True,
                                              padding_side="left")
    tokenizer.add_special_tokens({'pad_token': DEFINE_PAD_TOKEN})
    tokenizer.add_special_tokens({'sep_token': DEFINE_SEP_TOKEN})
    model.pad_token_id = tokenizer.pad_token_id
    model.pad_token = tokenizer.pad_token
    model.sep_token_id = tokenizer.sep_token_id
    model.sep_token = tokenizer.sep_token

    return model, tokenizer



def train():
    model, tokenizer = create_model_tokenizer(model_name)
    train_datasets, _ = create_prm_step_datasets(dataset_name, tokenizer, seq_length)
    collator = DataCollatorForSFT(tokenizer=tokenizer)
    peft_config = create_peft_prm_lm_head(True)
    model.enable_input_require_grads()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()


    training_args = TrainingArguments(
        output_dir=output_name,
        save_strategy='epoch',
        logging_steps=1,
        num_train_epochs=num_train_epochs,
        gradient_checkpointing=True,
        bf16=True,
        learning_rate=learning_rate,
        warmup_ratio=0.05,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        deepspeed=deepspeed_config_name,
        report_to='wandb',
        lr_scheduler_type='cosine',
        # max_steps=10,
    )

    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=train_datasets,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(output_name)
    tokenizer.save_pretrained(output_name)


if __name__ == "__main__":
    train()
