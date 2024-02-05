from responses import target
import torch
from datasets import load_dataset, load_from_disk
from trl import SFTTrainer
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from utils import ScriptArguments
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)

parser = HfArgumentParser(ScriptArguments)
train_args: ScriptArguments = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]

dataset_name = train_args.dataset_name
model_name = train_args.model_name
seq_length = train_args.seq_length
batch_size = train_args.batch_size
output_name = train_args.output_name
is_peft = train_args.use_QLora
is_use_flash_attention2 = train_args.use_flash_attention_2
deepspeed_config_name = train_args.deepspeed_config_name
num_train_epochs = train_args.num_train_epochs

def create_datasets(dataset_name, tokenizer):
    dataset = load_dataset(dataset_name,  split="train", num_proc=32) # for imdb
    # dataset = load_dataset(dataset_name,  split="train", num_proc=32).select(range(10000)) # for chinese continue training
    # print(len(dataset['text']))
    return dataset, None


def create_model_tokenizer(name):
    # QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    device_map = {"": Accelerator().local_process_index}
    print('device map: ', device_map)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        # torch_dtype=torch.bfloat16,
        use_flash_attention_2=True, # gpt 2 not support flash attention2
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            use_fast=True,
                                            trust_remote_code=True)

    return model, tokenizer


def create_peft(peft_flag):
    if peft_flag == False:
        return None
    else:
        # default peft lora is Q_Lora K_Lora
        peft_config = LoraConfig(
            r=32,
            lora_alpha=8,
            bias="none",
            task_type="CAUSAL_LM",
            # target_modules=["W_pack"], # for baichuan2
        )
        return peft_config


def formatting_func(example):
    return example['text']


def train():

    model, tokenizer = create_model_tokenizer(model_name)
    torch.distributed.barrier()

    tokenizer.pad_token = tokenizer.eos_token
    train_dataset, _ = create_datasets(dataset_name, tokenizer)
    torch.distributed.barrier()

    # peft
    peft_config = create_peft(is_peft)

    training_args = TrainingArguments(
        output_dir=output_name,
        push_to_hub=False,
        # save_strategy='epoch',
        logging_steps=1,
        num_train_epochs=1,
        gradient_checkpointing=True,
        bf16=True,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        deepspeed=deepspeed_config_name,
        report_to='wandb',
        lr_scheduler_type='cosine'
        # max_steps=10,
    )

    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=seq_length,
        peft_config=peft_config,
        packing=True,
        tokenizer=tokenizer,
        formatting_func=formatting_func,
        dataset_num_proc=16,
    )

    trainer.train()
    trainer.save_model(output_name)

if __name__ == "__main__":
    train()
