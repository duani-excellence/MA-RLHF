import torch
from transformers.utils import PaddingStrategy
from accelerate import Accelerator
from peft import LoraConfig, TaskType
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    HfArgumentParser,
)

DEFINE_EOS_TOKEN = '''</s>'''
DEFINE_BOS_TOKEN = '''<s>'''


def is_main_process():
    return torch.distributed.get_rank() == 0


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

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    return model, tokenizer


def create_peft(peft_flag: bool = False) -> LoraConfig:
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


def create_peft_reward_model(peft_flag: bool = False) -> LoraConfig:
    if peft_flag == False:
        return None
    else:
        # default peft lora is Q_Lora K_Lora
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=32,
            lora_alpha=8,
            bias="none",
            modules_to_save=["scores"],
        )
        return peft_config


@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_j = []
        features_k = []
        for feature in features:
            features_j.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            features_k.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})

    base_model_name: Optional[str] = field(default="", metadata={"help": "pretrained"})

    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})

    merged_model_name: Optional[str] = field(default="", metadata={"help": "lora + model"})

    output_name: Optional[str] = field(default="", metadata={"help": "n steps to save the model"})

    dataset_name: Optional[str] = field(
        default="", metadata={"help": "chinese medical english alpaca"}
    )

    deepspeed_config_name: Optional[str] = field(default="", metadata={"help": "ds.json"})

    prompt: Optional[str] = field(default="", metadata={"help": "for test generation"})

    learning_rate: Optional[float] = field(
        default=1.41e-5, metadata={"help": "todo: the learning rate,"}
    )

    seq_length: Optional[int] = field(default=512, metadata={"help": "context max length"})

    max_new_tokens: Optional[int] = field(default=128, metadata={"help": "max generate tokens"})

    output_max_length: Optional[int] = field(
        default=128, metadata={"help": "ppo maximum length for generation"}
    )

    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})

    batch_size: Optional[int] = field(default=8, metadata={"help": "the batch size"})

    ppo_epochs: Optional[int] = field(default=1, metadata={"help": "the number of ppo epochs"})

    num_train_epochs:  Optional[int] = field(default=1, metadata={"help": "train epochs "})

    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "gradient accumulation steps"}
    )

    early_stopping: Optional[bool] = field(
        default=False, metadata={"help": "whether to early stop"}
    )

    target_kl: Optional[float] = field(
        default=0.1, metadata={"help": "kl target for early stopping"}
    )

    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})

    use_QLora: Optional[bool] = field(default=True, metadata={"help": "todo optional"})

    use_flash_attention_2: Optional[bool] = field(
        default=True, metadata={"help": "gpt2 no flash attention2"}
    )

def format_prompt_answer(question, answer):
    '''for generation'''
    return f"###Question: {question}\n###Answer: {answer} {DEFINE_EOS_TOKEN}"


def format_prompt(question):
    return f"###Question: {question}\n###Answer: "


# medical finetune data haven't 'input', only has 'instruction'
def formatting_finetune_func(example):
    text = f"###Question: {example['instruction']}\n###Answer: {example['output']} {DEFINE_EOS_TOKEN}"
    return text


def formatting_reward_func(example):
    text = f"###Question: {example['question']}\n###Answer: {example['response_rejected']} {DEFINE_EOS_TOKEN}"
    return text


def formatting_alpaca_func(example):
    return f"###Question: {example['instruction']} {example['input']}\n###Answer: {example['output']} {DEFINE_EOS_TOKEN}"



def formatting_alpaca_chinese_func(example):
    return f"###Question: {example['instruction_zh']} {example['input_zh']}\n###Answer: {example['output_zh']}{DEFINE_EOS_TOKEN}"
