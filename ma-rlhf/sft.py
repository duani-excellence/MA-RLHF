import torch
from datasets import load_dataset, load_from_disk
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset
from accelerate import Accelerator
from peft import LoraConfig


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
    formatting_finetune_func,
    formatting_reward_func,
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
is_use_flash_attention2 = train_args.use_flash_attention_2


def create_datasets(dataset_name, dataset_sub_name):
    print(dataset_name)
    print(dataset_sub_name)
    # dataset = load_dataset(dataset_name, dataset_sub_name, trust_remote_code=True)
    dataset = load_from_disk(dataset_name)
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
        # use_flash_attention_2=True # gpt 2 not support flash attention2
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

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
        )
        return peft_config


def create_sft_datasets(datasets, tokenizer, format_func, seq_length=512):
    train_data = datasets["train"]
    valid_data = datasets["test"]

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=format_func,
        infinite=True,
        seq_length=seq_length,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=format_func,
        infinite=False,
        seq_length=seq_length,
    )
    return train_dataset, valid_dataset


# # medical finetune data haven't 'input', only has 'instruction'
# def formatting_finetune_func(example):
#     text = f"### Question: {example['instruction']}\n ### Answer: {example['output']}{DEFINE_EOS_TOKEN}"
#     return text

# def formatting_reward_func(example):
#     text = f"### Question: {example['question']}\n ### Answer: {example['response_rejected']}{DEFINE_EOS_TOKEN}"
#     return text


def train():
    model, tokenizer = create_model_tokenizer(model_name)
    # torch.distributed.barrier()

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.eos_token = DEFINE_EOS_TOKEN
    datasets, _ = create_datasets(dataset_name, dataset_sub_name)

    format_fun = None
    if dataset_sub_name == 'finetune':
        format_fun = formatting_finetune_func
    elif dataset_sub_name == 'reward':
        format_fun = formatting_reward_func
    else:
        format_fun = None

    train_datasets, val_datasets = create_sft_datasets(datasets, tokenizer, format_fun, seq_length)

    # peft
    peft_config = create_peft(is_peft)

    # output_name = './'
    # print(output_name)
    training_args = TrainingArguments(
        output_dir=output_name,
        save_strategy='steps',
        logging_steps=1,
        num_train_epochs=2,
        gradient_checkpointing=True,
        bf16=True,
        learning_rate=5e-5,
        warmup_ratio=0.05,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        deepspeed=deepspeed_config_name,
        report_to='wandb',
    )

    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=train_datasets,
        eval_dataset=val_datasets,
        # dataset_text_field="text",
        max_seq_length=seq_length,
        peft_config=peft_config,
        packing=True,
        tokenizer=tokenizer,
        # formatting_func=formatting_reward_func,
    )
    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(output_name)


if __name__ == "__main__":
    # with torch.autocast("cuda"):
    train()
