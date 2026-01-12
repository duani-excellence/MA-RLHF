# use 3090x2, 17.42min training 1 epochs alpaca
# deepspeed sft_test.py
# accelerate launch --num_processes=2 sft_test.py

from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
import torch

DEFINIED_SYSTEM_PROMPT = '你是小冬瓜智能体,请安全详细回答用户 USER 的问题'

def create_model(name):
    """获取 QLoRA 量化模型 """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
            name,
            quantization_config=bnb_config, # 注释则
            device_map='auto',
            # attn_implementation = "flash_attention_2",
            trust_remote_code=True,
            dtype=torch.bfloat16,
            use_cache=False,
        )
    tokenizer =  AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')
    return model, tokenizer

def get_lora_config():
    peft_config = LoraConfig(
            r=64,
            lora_alpha=8,
            bias="none",
            # lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'lm_head'],
        )
    return peft_config

def get_dataset_alpaca():
    dataset = load_dataset('tatsu-lab/alpaca')
    def map_cat_inst_input(example):
        example['messages'] = [
            {'role':'system', 'content':DEFINIED_SYSTEM_PROMPT},
            {'role':'user', 'content': example['instruction']+example['input']},
            {'role':'assistant', 'content': example['output']},
        ]
        return example
    dataset_alpaca = dataset.map(map_cat_inst_input,
                                remove_columns=["instruction", "input", "output", "text"])
    return dataset_alpaca


def train(model_name, output_name):
    model, tokenizer = create_model(name=model_name)
    peft_config = get_lora_config()
    dataset_alpaca = get_dataset_alpaca()

    config = SFTConfig(
        output_dir=output_name,
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 4,
        max_length = 512,
        # max_steps = 10, # 调试环境时, 开启这个
        num_train_epochs=1, # 调试环境时, 关闭这个
        bf16=True,
        deepspeed='./ds.json',
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset_alpaca['train'],
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(output_name)
    
if __name__ == "__main__":
    model_name = 'Qwen/Qwen3-0.6B'
    output_name = './output/Qwen3-0.6B-SFT'
    train(model_name = model_name,
          output_name = output_name)