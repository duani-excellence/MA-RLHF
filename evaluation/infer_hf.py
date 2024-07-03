# torchrun --nnodes=1 --nproc_per_node=16  infer_hf.py  \
# 	--model_name='/mnt/output/sft_full' \
# 	--output_name='/mnt/output/result/tmp'  \
# 	--max_output_tokens=512 \
# 	--batch_size=2
# test in v100 : 30B model generate with > 20G VRAM

import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import accelerate
import json
import time
import argparse
import os

torch.manual_seed(42)

DEFINE_BOS_TOKEN = '''<s>'''
DEFINE_EOS_TOKEN = '''</s>'''
DEFINE_PAD_TOKEN = '''<pad>'''
SYSTEM_PROMPT = '''You are a robot named "MA-RLHF", you are always friendly and answer questions。'''


def format_prompt_answer(question, answer):
    '''for generation'''
    return f"###System: {SYSTEM_PROMPT}\n###Question: {question}\n###Answer: {answer} {DEFINE_EOS_TOKEN}"


def format_prompt(question):
    return f"###System: {SYSTEM_PROMPT}\n###Question: {question}\n###Answer: "


def load_model(model_path, rank):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # torch_dtype=torch.float16,
        use_flash_attention_2=False,
        trust_remote_code=True,
        quantization_config=bnb_config,
        # load_in_4bit=True,
        device_map=f'cuda:{rank}',
        # device_map='auto',
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding_side='left', trust_remote_code=True,)

    return model, tokenizer


def create_dataset(dataset_name, tokenizer):
    datasets = load_dataset(dataset_name, split='test')

    # columns = datasets.features
    def process(examples):
        prompt = format_prompt(examples['prompt'])
        # inputs = tokenizer([prompt], return_tensors='pt')
        inputs = tokenizer([prompt])
        examples['input_ids'] = inputs['input_ids'][0]
        examples['attention_mask'] = inputs['attention_mask'][0]
        return examples

    datasets = datasets.map(
        process,
        # batched=True,
        num_proc=24,
        remove_columns=['prompt', 'response_0', 'response_1', 'is_response_0_safe', 'is_response_1_safe',
                        'better_response_id', 'safer_response_id'],
    )

    datasets = datasets.filter(lambda x: len(x["input_ids"]) < 512)
    return datasets


def generate_dp(model_name, output_name, max_output_tokens, batch_size):

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)  # 没有这句话，cuda0将会炸

    if rank == 0:
        if not os.path.exists(output_name):
            os.makedirs(output_name)
            print(f"Folder '{output_name}' created successfully.")
        else:
            print(f"Folder '{output_name}' already exists.")

    dist.barrier()

    model, tokenizer = load_model(model_name, rank)

    dist.barrier()

    dataset = create_dataset('PKU-Alignment/PKU-SafeRLHF-30K', tokenizer)

    print(dataset)
    sampler = DistributedSampler(dataset)
    data_collator = DataCollatorWithPadding(
        tokenizer, max_length=512, padding=True)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            sampler=sampler, collate_fn=data_collator)

    step = 0
    step_all = len(dataloader)
    result_all = []
    # print(len(dataloader))

    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    dist.barrier()

    # time.sleep(100)

    for x in dataloader:
        torch.cuda.empty_cache()
        if rank == 0:
            print('-'*50)
            print(f'{step}/{step_all}')
            print('-'*50)
        x['input_ids'] = torch.tensor(
            x['input_ids'], dtype=torch.long).to(rank)
        x['attention_mask'] = torch.tensor(
            x['attention_mask'], dtype=torch.bool).to(rank)
        with torch.no_grad():
            y = model.generate(
                **x, max_new_tokens=max_output_tokens, do_sample=False)
        result = tokenizer.batch_decode(y, skip_special_tokens=True)
        result_all.extend(result)
        if rank == 0:
            for str in result:
                print('*'*50)
                print(f'[{rank}]:{str}')
        step = step+1
        # break

    current_result_file = output_name + f'/result_{rank}.json'
    with open(current_result_file, 'w') as file:
        json.dump(result_all, file)

    dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_name',  type=str, default='', required=True,
                        help='model name path')
    parser.add_argument('--output_name', type=str, default='', required=True,
                        help='output path')
    parser.add_argument('--max_output_tokens', type=int, default=100,
                        help='generation tokens')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='generation tokens')

    args = parser.parse_args()
    print(args)

    generate_dp(args.model_name,
                args.output_name,
                args.max_output_tokens,
                args.batch_size)
