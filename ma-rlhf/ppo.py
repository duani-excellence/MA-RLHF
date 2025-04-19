'''
this code required trl>0.11.0 and NOT SUPPORT multi-adapter LoRA training
'''

import re
import torch
import os
from datasets import load_dataset, load_from_disk
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer, PPOConfig
from trl.core import LengthSampler
from transformers import  (
    AutoTokenizer, 
    BitsAndBytesConfig, 
    HfArgumentParser, 
    GenerationConfig, 
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
from accelerate import Accelerator
from utils import (
    create_model_tokenizer,
    create_peft,
    is_main_process,
    ScriptArguments,
    DEFINE_EOS_TOKEN,
    DEFINE_PAD_TOKEN,
    format_prompt,
    SYSTEM_PROMPT,
)
import time

os.environ["WANDB_PROJECT"] = "ma-rlhf"
os.environ["WANDB_RUN_NAME"] = "ppo"


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

gradient_accumulation_steps = train_args.gradient_accumulation_steps

def create_model_tokenizer(name, rm_model_name, peft_config):
    # QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        # bnb_4bit_use_double_quant=True,
    )

    device_map = {"": Accelerator().local_process_index}
    print('device map: ', device_map)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        # use_fast=True,
        trust_remote_code=True,
    )
    tokenizer.add_special_tokens({'pad_token': DEFINE_PAD_TOKEN})

     # generation config
    generation_kwargs = {
        "min_length": -1,
        "max_new_tokens": output_max_length,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "forced_eos_token_id": tokenizer.eos_token_id, # class ForcedEOSTokenLogitsProcessor(LogitsProcessor) from transformers
        # "forced_eos_token_id": True,
    }

    value_model = AutoModelForSequenceClassification.from_pretrained(
        rm_model_name, trust_remote_code=True, num_labels=1
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        rm_model_name, trust_remote_code=True, num_labels=1
    )
    policy_model = AutoModelForCausalLM.from_pretrained(
        name, trust_remote_code=True
    )

    # peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            name, trust_remote_code=True
        )
    else:
        ref_model = None

    

    return policy_model, value_model, reward_model, ref_model, tokenizer


def create_dataset(dataset_name, tokenizer):

    datasets = load_dataset(dataset_name, split='train')

    def preprocess_function(examples):
        outputs = tokenizer(
            examples['prompt'],
            padding=False,
        )
        return {"input_ids": outputs["input_ids"]}

    func = preprocess_function

    datasets = datasets.map(
        func,
        batched=True,
        num_proc=24,
        remove_columns=datasets.column_names,
    )

    datasets = datasets.filter(lambda x: len(x["input_ids"]) < seq_length, batched=False)
    return datasets


def train():
    peft_config = create_peft(is_peft)
    policy_model, value_model, reward_model, ref_model, tokenizer = create_model_tokenizer(
        model_name, rm_model_name, peft_config
    )  # model is sequence classification

    dataset = create_dataset(dataset_name, tokenizer)
    print(dataset)

    # output_length_sampler = LengthSampler(128, output_max_length)

    config = PPOConfig(
        learning_rate=1e-5,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_ppo_epochs=ppo_epochs,
        seed=0,
        max_grad_norm=1.0,  # fix generate nan
        model_adapter_name=model_name,
        ref_adapter_name=model_name,
        reward_model_path=rm_model_name,
    )

    trainer = PPOTrainer(
        config,
        model=policy_model,
        ref_model=ref_model,  # share parameters
        value_model=value_model,
        reward_model=reward_model,
        train_dataset=dataset,
        eval_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()


    # 训练时把trl `ppo_trainer.py` 以下函数修改，方可保存
    # def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
    #     backup_model = self.model
    #     # self.model = self.model.policy # 删除这行
    #     self.model = self.model.module.policy  # 修改成这行
    trainer.save_model(output_name)

if __name__ == "__main__":
    train()
