import torch
from datasets import load_dataset, load_from_disk
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset
from accelerate import Accelerator
from peft import LoraConfig
from unsloth import FastLanguageModel


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
    formatting_alpaca_func,
    # formatting_alpaca_chinese_func,
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
dataset_sub_name = None
num_train_epochs = train_args.num_train_epochs


def create_datasets(dataset_name, dataset_sub_name):
    # print(dataset_name)
    # print(dataset_sub_name)
    dataset = load_dataset(dataset_name)
    # print(len(dataset['text']))
    return dataset, None


def create_model_tokenizer(name):
    device_map = {"": Accelerator().local_process_index}
    print('device map: ', device_map)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,  # Supports Llama, Mistral - replace this!
        max_seq_length=seq_length,
        dtype=None,
        load_in_4bit=True,
        device_map=device_map,
    )

    return model, tokenizer

def create_peft(peft_flag):
    if peft_flag == False:
        return None
    else:
        peft_config = LoraConfig(
            r=32,
            lora_alpha=8,
            bias="none",
            task_type="CAUSAL_LM",
            # target_modules=["W_pack"],
        )
        return peft_config


def create_sft_datasets(datasets, tokenizer, format_func, seq_length=512):
    train_dataset = datasets["train"]

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_dataset,
        formatting_func=format_func,
        infinite=True,
        seq_length=seq_length,
    )

    return train_dataset, None

def train():
    model, tokenizer = create_model_tokenizer(model_name)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.eos_token = DEFINE_EOS_TOKEN
    datasets, _ = create_datasets(dataset_name, dataset_sub_name)

    # format_fun = None
    # if dataset_sub_name == 'finetune':
    #     format_fun = formatting_finetune_func
    # elif dataset_sub_name == 'reward':
    #     format_fun = formatting_reward_func
    # else:
    #     format_fun = None
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
        learning_rate=2e-5,
        warmup_ratio=0.03,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
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
        # formatting_func=formatting_alpaca_func,
    )
    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(output_name)


if __name__ == "__main__":
    # with torch.autocast("cuda"):
    train()