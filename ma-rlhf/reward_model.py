import os
import deepspeed
from datasets import load_dataset, load_from_disk
from trl import RewardTrainer, RewardConfig
import re
import torch
import evaluate
from accelerate import Accelerator
import numpy as np
from utils import (
    create_peft_reward_model,
    ScriptArguments,
    DEFINE_EOS_TOKEN,
    DEFINE_PAD_TOKEN,
    format_prompt_answer,
    SYSTEM_PROMPT,
)
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoModelForSequenceClassification,
)
os.environ["WANDB_PROJECT"] = "ma-rlhf"
os.environ["WANDB_RUN_NAME"] = "reward_model"

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
num_train_epochs = train_args.num_train_epochs
gradient_accumulation_steps = train_args.gradient_accumulation_steps
learning_rate = train_args.learning_rate

# accuracy = evaluate.load("accuracy")
# def compute_metrics(eval_pred):
#     predictions, _ = eval_pred
#     # Here, predictions is rewards_j and rewards_k.
#     # We want to see how much of the time rewards_j > rewards_k.
#     predictions = np.argmax(predictions, axis=0)
#     labels = np.zeros(predictions.shape)
#     return accuracy.compute(predictions=predictions, references=labels)


def create_model_tokenizer(name):
    # QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    device_map = {"": Accelerator().local_process_index}
    print('device map: ', device_map)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map, # 70b use auto
        num_labels=1
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    tokenizer.add_special_tokens({'pad_token': DEFINE_PAD_TOKEN})
    model.pad_token_id = tokenizer.pad_token_id
    model.pad_token = tokenizer.pad_token

    return model, tokenizer


tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, max_length=512)
tokenizer.add_special_tokens({'pad_token': DEFINE_PAD_TOKEN})

# for prompt, chosen, rejected
def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for question, response_j, response_k in zip(
        examples["question"], examples["response_chosen"], examples["response_rejected"]
    ):

        str_chosen = format_prompt_answer(question, response_j)
        str_rejected = format_prompt_answer(question, response_k)

        tokenized_j = tokenizer(
            str_chosen,
            truncation=True,
            padding='longest'
        )
        tokenized_k = tokenizer(
            str_rejected,
            truncation=True,
            padding='longest'
        )

        new_examples["input_ids_chosen"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_k["attention_mask"])

    return new_examples



def preprocess_function_safe_rlhf(examples):
    '''
    reference: https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF-10K
    '''
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for question, response_0, response_1, safe_0, safe_1, better_id, safer_id in zip(
        examples["prompt"], examples["response_0"], examples["response_1"],
        examples['is_response_0_safe'], examples['is_response_1_safe'],
        examples['better_response_id'], examples['safer_response_id'],
    ):
        response_chosen = None
        response_rejected = None

        if safe_0 == True and safe_1 == True :
            response_chosen = response_0 if better_id == 0 else response_1
            response_rejected = response_1 if better_id == 0 else response_0
        elif safe_0 == True and safe_1 == False:
            response_chosen = response_0
            response_rejected = response_1
        elif safe_0 == False and safe_1 == True:
            response_chosen = response_1
            response_rejected = response_0
        elif safe_0 == False and safe_1 == False: # TODO should filter both unsafe examples
            response_chosen = response_0 if better_id == 0 else response_1
            response_rejected = response_1 if better_id == 0 else response_0

        str_chosen = format_prompt_answer(question, response_chosen)
        str_rejected = format_prompt_answer(question, response_rejected)

        tokenized_chosen = tokenizer(
            str_chosen,
            truncation=True,
            padding='longest',
        )
        tokenized_rejected = tokenizer(
            str_rejected,
            truncation=True,
            padding='longest',
        )

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples



# Anthropic/hh-rlhf
# chosen, rejected
def preprocess_function_hhrlhf(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for prompt_chosen, prompt_rejected in zip(examples["chosen"], examples["rejected"]):

        prompt_chosen = re.sub(r'\n\nHuman:', '\n###Question:', prompt_chosen)
        prompt_chosen = re.sub(r'\n\nAssistant:', '\n###Answer:', prompt_chosen)
        prompt_chosen = prompt_chosen[1:] # ignore first \n
        prompt_rejected = re.sub(r'\n\nHuman:', '\n###Question:', prompt_rejected)
        prompt_rejected = re.sub(r'\n\nAssistant:', '\n###Answer:', prompt_rejected)
        prompt_rejected = prompt_rejected[1:] # ignore first \n

        prompt_chosen = f'###System: {SYSTEM_PROMPT}\n{prompt_chosen}'
        prompt_rejected = f'###System: {SYSTEM_PROMPT}\n{prompt_rejected}'

        tokenized_j = tokenizer(
            prompt_chosen,
            truncation=True,
        )
        tokenized_k = tokenizer(
            prompt_rejected,
            truncation=True,
        )

        new_examples["input_ids_chosen"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_k["attention_mask"])

    return new_examples


def create_reward_model_datasets(datasets_name, dataset_sub_name, tokenizer):
    train_dataset = load_dataset(datasets_name, split='train')
    # eval_dataset = load_dataset(datasets_name, split='test')

    func = None
    if 'hh-rlhf' in dataset_name:
        func = preprocess_function_hhrlhf
    elif 'SafeRLHF' in dataset_name:
        func = preprocess_function_safe_rlhf
    else:
        func = preprocess_function


    train_dataset = train_dataset.map(
        func,
        batched=True,
        num_proc=24,
    )

    torch.distributed.barrier()
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= seq_length
        and len(x["input_ids_rejected"]) <= seq_length
    )
    torch.distributed.barrier()

    # eval_dataset = eval_dataset.map(
    #     preprocess_function_hhrlhf,
    #     batched=True,
    #     num_proc=8,
    # )

    # torch.distributed.barrier()
    # eval_dataset = eval_dataset.filter(
    #     lambda x: len(x["input_ids_chosen"]) <= seq_length
    #     and len(x["input_ids_rejected"]) <= seq_length
    # )

    # torch.distributed.barrier()

    # return train_dataset, eval_dataset
    return train_dataset, None


def train():
    model, tokenizer = create_model_tokenizer(model_name)  # model is sequence classification
    train_datasets, test_datasets = create_reward_model_datasets(dataset_name, None, tokenizer)

    # PEFT
    peft_config = create_peft_reward_model(is_peft)

    # ZERO stage3 use config like # https://github.com/huggingface/trl/issues/835
    reward_config = RewardConfig(
        output_dir=output_name,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        report_to="wandb",
        warmup_ratio=0.01,
        remove_unused_columns=True,
        optim="adamw_torch",
        logging_steps=1,
        max_length=seq_length,
        deepspeed=deepspeed_config_name,
        bf16=True,
        lr_scheduler_type='cosine',
        # evaluation_strategy="steps",
        # eval_steps=100,
        evaluation_strategy='no',
        # max_steps=10,
    )

    trainer = RewardTrainer(
        model,
        args=reward_config,
        train_dataset=train_datasets,
        # eval_dataset=test_datasets,
        # compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(output_name)


if __name__ == "__main__":
    train()
