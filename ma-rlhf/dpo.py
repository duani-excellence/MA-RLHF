import os
import numpy as np
from datasets import load_dataset, load_from_disk, concatenate_datasets
from trl import DPOTrainer
import re
import torch
import evaluate
from accelerate import Accelerator
from utils import (
    create_peft_reward_model,
    ScriptArguments,
    DEFINE_EOS_TOKEN,
    create_peft,
)
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    TrainingArguments
)

os.environ["WANDB_PROJECT"] = "ma-rlhf"
os.environ["WANDB_RUN_NAME"] = "dpo"

parser = HfArgumentParser(ScriptArguments)
train_args: ScriptArguments = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]

dataset_name = train_args.dataset_name
model_name = train_args.model_name
deepspeed_config_name = train_args.deepspeed_config_name
output_max_length = train_args.output_max_length
seq_length = train_args.seq_length
batch_size = train_args.batch_size
output_name = train_args.output_name
is_peft = train_args.use_QLora
is_use_flash_attention2 = train_args.use_flash_attention_2
num_train_epochs = train_args.num_train_epochs
beta = 0.1 # default
gradient_accumulation_steps = train_args.gradient_accumulation_steps
learning_rate = train_args.learning_rate


accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    # Here, predictions is rewards_j and rewards_k.
    # We want to see how much of the time rewards_j > rewards_k.
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)


def create_model_tokenizer(name):
    # QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        # bnb_4bit_use_double_quant=True,
    )

    device_map = {"": Accelerator().local_process_index}
    print('device map: ', device_map)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map=device_map,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=seq_length,
        trust_remote_code=True,)

    tokenizer.eos_token = DEFINE_EOS_TOKEN
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.eos_token = DEFINE_EOS_TOKEN
tokenizer.pad_token = tokenizer.eos_token


# Anthropic/hh-rlhf
# chosen, rejected
def preprocess_function_hhrlhf(examples):
    new_examples = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }

    for prompt_chosen, prompt_rejected in zip(
        examples["chosen"], examples["rejected"]
    ):
        prompt_chosen = re.sub(r'\n\nHuman:', '\n###Question:', prompt_chosen)
        prompt_chosen = re.sub(r'\n\nAssistant:', '\n###Answer:', prompt_chosen)
        prompt_chosen = prompt_chosen[1:] # ignore first \n
        prompt_rejected = re.sub(r'\n\nHuman:', '\n###Question:', prompt_rejected)
        prompt_rejected = re.sub(r'\n\nAssistant:', '\n###Answer:', prompt_rejected)
        prompt_rejected = prompt_rejected[1:] # ignore first \n

        prompt_question = prompt_chosen.rsplit('\n###Answer:',1)[0] + '\n###Answer:'
        response_chosen = prompt_chosen.rsplit('\n###Answer:',1)[1] + ' ' + DEFINE_EOS_TOKEN
        response_rejected = prompt_rejected.rsplit('\n###Answer:',1)[1] + ' ' + DEFINE_EOS_TOKEN
        # print(f'[prompt]:{prompt_question}\n[chosen]{response_chosen}\n[rejected]{response_rejected}')

        new_examples['prompt'].append(prompt_question)
        new_examples['chosen'].append(response_chosen)
        new_examples['rejected'].append(response_rejected)

    return new_examples


def create_dpo_datasets(datasets_name, dataset_sub_name, tokenizer):
    train_dataset = load_dataset(datasets_name, split='train')

    # merge other datasets and shuffle
    # train_dataset2 = load_dataset('Unified-Language-Model-Alignment/Anthropic_HH_Golden', split='train')
    # train_dataset = concatenate_datasets([train_dataset, train_dataset2])
    # train_dataset = train_dataset.shuffle(seed=42)

    eval_dataset = load_dataset(datasets_name, split='test')
    # eval_dataset2 = load_dataset('Unified-Language-Model-Alignment/Anthropic_HH_Golden', split='test')
    # eval_dataset = concatenate_datasets([eval_dataset, eval_dataset2])
    # eval_dataset = eval_dataset.shuffle(seed=42)

    train_dataset = train_dataset.map(
        preprocess_function_hhrlhf,
        batched=True,
        num_proc=16,
    )

    train_dataset = train_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= seq_length
        and len(x["prompt"]) + len(x["rejected"]) <= seq_length
    )

    eval_dataset = eval_dataset.map(
        preprocess_function_hhrlhf,
        batched=True,
        num_proc=16,
    )

    eval_dataset = eval_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= seq_length
        and len(x["prompt"]) + len(x["rejected"]) <= seq_length
    )

    return train_dataset, eval_dataset


def train():
    model, tokenizer = create_model_tokenizer(model_name)  # model is sequence classification
    train_datasets, test_datasets = create_dpo_datasets(
        dataset_name, None, tokenizer
    )

    # PEFT
    peft_config = create_peft(is_peft)

    training_args = TrainingArguments(
        output_dir=output_name,
        # push_to_hub=False,
        # save_strategy='epoch',
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
        # max_steps=200,
    )

    trainer = DPOTrainer(
        model,
        None,
        args=training_args,
        beta=beta,
        train_dataset=train_datasets,
        eval_dataset=test_datasets,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length= output_max_length,
        max_length=seq_length,
        max_target_length=output_max_length,
    )

    trainer.train()
    trainer.save_model(output_name)


if __name__ == "__main__":
    train()
