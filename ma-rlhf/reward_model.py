from datasets import load_dataset, load_from_disk
from trl import RewardTrainer, RewardConfig
import re
import torch
import evaluate
from accelerate import Accelerator
from utils import (
    create_peft_reward_model,
    ScriptArguments,
    DEFINE_EOS_TOKEN,
)
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoModelForSequenceClassification,
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
num_train_epochs = train_args.num_train_epochs


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
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    device_map = {"": Accelerator().local_process_index}
    print('device map: ', device_map)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, quantization_config=bnb_config, device_map=device_map, num_labels=1
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.eos_token = DEFINE_EOS_TOKEN
    # https://stackoverflow.com/questions/68084302/assertionerror-cannot-handle-batch-sizes-1-if-no-padding-token-is-defined
    model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.eos_token = DEFINE_EOS_TOKEN

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
        tokenized_j = tokenizer(
            f"###Question:{question}\n###Answer:{response_j}{tokenizer.eos_token}",
            truncation=True,
        )
        tokenized_k = tokenizer(
            f"###Question:{question}\n###Answer:{response_k}{tokenizer.eos_token}",
            truncation=True,
        )

        new_examples["input_ids_chosen"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_k["attention_mask"])

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

        prompt_chosen = re.sub(r'Human:', '###Question:', prompt_chosen)
        prompt_chosen = re.sub(r'Assistant:', '\n###Answer:', prompt_chosen)
        prompt_rejected = re.sub(r'Human:', '###Question:', prompt_rejected)
        prompt_rejected = re.sub(r'Assistant:', '\n###Answer:', prompt_rejected)

        tokenized_j = tokenizer(
            f"{prompt_chosen}{tokenizer.eos_token}",
            truncation=True,
        )
        tokenized_k = tokenizer(
            f"{prompt_rejected}{tokenizer.eos_token}",
            truncation=True,
        )

        new_examples["input_ids_chosen"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_k["attention_mask"])

    return new_examples


def create_reward_model_datasets(datasets_name, dataset_sub_name, tokenizer):
    train_dataset = load_dataset(datasets_name, split='train')
    eval_dataset = load_dataset(datasets_name, split='test')
    # print(train_dataset)
    # print(eval_dataset)

    train_dataset = train_dataset.map(
        preprocess_function_hhrlhf,
        batched=True,
        num_proc=8,
    )

    torch.distributed.barrier()
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= seq_length
        and len(x["input_ids_rejected"]) <= seq_length
    )
    torch.distributed.barrier()

    eval_dataset = eval_dataset.map(
        preprocess_function_hhrlhf,
        batched=True,
        num_proc=8,
    )

    torch.distributed.barrier()
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= seq_length
        and len(x["input_ids_rejected"]) <= seq_length
    )

    torch.distributed.barrier()

    return train_dataset, eval_dataset


def train():
    model, tokenizer = create_model_tokenizer(model_name)  # model is sequence classification
    train_datasets, test_datasets = create_reward_model_datasets(dataset_name, None, tokenizer)

    # PEFT
    peft_config = create_peft_reward_model(is_peft)

    reward_config = RewardConfig(
        output_dir=output_name,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        learning_rate=2e-5,
        report_to="wandb",
        warmup_ratio=0.01,
        remove_unused_columns=True,
        optim="adamw_torch",
        logging_steps=1,
        max_length=seq_length,
        deepspeed=deepspeed_config_name,
        bf16=True,
        lr_scheduler_type='cosine',
        evaluation_strategy="steps",
        eval_steps=100,
        # max_steps=10,
    )

    trainer = RewardTrainer(
        model,
        args=reward_config,
        train_dataset=train_datasets,
        eval_dataset=test_datasets,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(output_name)


if __name__ == "__main__":
    train()
