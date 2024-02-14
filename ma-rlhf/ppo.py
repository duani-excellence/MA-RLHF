import re
import torch
from datasets import load_dataset, load_from_disk
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer, PPOConfig
from trl.core import LengthSampler
from transformers import AutoTokenizer, BitsAndBytesConfig, HfArgumentParser
from accelerate import Accelerator
from utils import (
    create_model_tokenizer,
    create_peft,
    is_main_process,
    ScriptArguments,
    DEFINE_EOS_TOKEN,
)

# class MyPPOTrainer(PPOTrainer):

parser = HfArgumentParser(ScriptArguments)
train_args: ScriptArguments = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]

dataset_name = train_args.dataset_name
model_name = train_args.model_name
rm_model_name = train_args.reward_model_name
deepspeed_config_name = train_args.deepspeed_config_name
batch_size = train_args.batch_size
mini_batch_size = train_args.mini_batch_size
ppo_epochs = train_args.ppo_epochs
output_max_length = train_args.output_max_length
seq_length = train_args.seq_length
output_name = train_args.output_name
is_peft = train_args.use_QLora
is_use_flash_attention2 = train_args.use_flash_attention_2
# num_train_epochs = train_args.num_train_epochs
gradient_accumulation_steps = train_args.gradient_accumulation_steps

def create_model_tokenizer(name, rm_model_name, peft_config):
    # QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    device_map = {"": Accelerator().local_process_index}
    print('device map: ', device_map)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        name,
        quantization_config=bnb_config,
        peft_config=peft_config,
        reward_adapter=rm_model_name,
        device_map=device_map,
        use_flash_attention_2=True,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.eos_token = DEFINE_EOS_TOKEN
    # https://stackoverflow.com/questions/68084302/assertionerror-cannot-handle-batch-sizes-1-if-no-padding-token-is-defined
    model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


def create_dataset(dataset_name, tokenizer):

    datasets = load_dataset(dataset_name, split='train')

    # template: ###Question: {question}\n ###Answer: {response_j}{tokenizer.eos_token}
    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in examples["question"]:
            query = f"###Question:{question}\n###Answer:"
            tokenized_question = tokenizer(query, truncation=True, return_tensors='pt')
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"][0])
        return new_examples

    def preprocess_function_hhrlhf(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for prompt_chosen in examples["chosen"]:

            # format hh-rlhf dataset for PPO
            prompt_chosen = prompt_chosen.rsplit('Assistant:',1)[0]
            prompt_chosen = re.sub(r'Human:', '###Question:', prompt_chosen)
            prompt_chosen = re.sub(r'Assistant:', '###Answer:', prompt_chosen)
            query = prompt_chosen + '###Answer:'

            # TODO:truncation Answer Process
            tokenized_question = tokenizer(query, return_tensors='pt')
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"][0])
        return new_examples


    datasets = datasets.map(
        preprocess_function_hhrlhf,
        batched=True,
        num_proc=8,
        remove_columns=datasets.column_names,
    )

    datasets = datasets.filter(lambda x: len(x["input_ids"]) < seq_length, batched=False)
    datasets.set_format(type="torch")
    return datasets


def collator(examples):
    batch = {'query': [], 'input_ids': []}
    for example in examples:
        batch['query'].append(example['query'])
        batch['input_ids'].append(torch.tensor(example['input_ids'], dtype=torch.long))
    return batch

def train():
    peft_config = create_peft(is_peft)
    model, tokenizer = create_model_tokenizer(
        model_name, rm_model_name, peft_config
    )  # model is sequence classification

    dataset = create_dataset(dataset_name, tokenizer)
    print(dataset)

    # generation config
    generation_kwargs = {
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 1.2,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        # "forced_eos_token_id": True,
    }
    output_length_sampler = LengthSampler(128, output_max_length)

    config = PPOConfig(
        log_with='wandb',
        learning_rate=2e-5,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optimize_cuda_cache=True,
        early_stopping=True,
        target_kl=0.1,
        ppo_epochs=ppo_epochs,
        seed=0,
        init_kl_coef=0.2,
        adap_kl_ctrl=True,
        max_grad_norm=1.0,  # fix generate nan
    )

    trainer = PPOTrainer(
        config,
        model,
        ref_model=None,  # share parameters
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )

    reward_baseline = 0.0
    save_freq = 50

    # for epoch, batch in enumerate(trainer.dataloader):
    for epoch, batch in enumerate(trainer.dataloader):
        if epoch >= config.total_ppo_epochs:
            break
        question_tensors = batch["input_ids"]
        response_tensors = trainer.generate(
            question_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        rm_model = trainer.accelerator.unwrap_model(trainer.model)
        raw_rewards = []
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt').to(trainer.accelerator.device)
            score = rm_model.compute_reward_score(**inputs)[0,-1,0]
            raw_rewards.append(score)
        rewards = raw_rewards

        ## PPO Step
        stats = trainer.step(question_tensors, response_tensors, rewards)
        trainer.log_stats(stats, batch, rewards)

        if is_main_process():
            print(texts[0])
            print(rewards)
            print(f"step:{epoch}/all:{len(trainer.dataloader)},loss:{stats['ppo/loss/total']},mean_scores:{stats['ppo/mean_scores']}" )

        if save_freq and epoch and epoch % save_freq == 0:
            trainer.save_pretrained(f'{output_name}_{epoch}')
            print(f'{output_name}_{epoch}')
            # break
    trainer.save_pretrained(output_name)

if __name__ == "__main__":
    train()
