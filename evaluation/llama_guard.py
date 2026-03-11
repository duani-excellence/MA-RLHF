# no debug
import os
import re
import torch
from datasets import load_dataset, load_from_disk
import datasets
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
)
# from torch.distributed import DistributedSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import accelerate
import json
import time
# from tqdm import tqdm

# dist.init_process_group("nccl")
# rank = dist.get_rank()

import argparse

torch.manual_seed(42)


# # 定义文件夹路径
# folder_path = 'path/to/your/folder'

# # 如果文件夹不存在，则创建文件夹
# if not os.path.exists(folder_path):
#     os.makedirs(folder_path)
#     print(f"Folder '{folder_path}' created successfully.")
# else:
#     print(f"Folder '{folder_path}' already exists.")


def format_prompt_answer(question, answer):
    '''for generation'''
    return f"###System: {SYSTEM_PROMPT}\n###Question: {question}\n###Answer: {answer} {DEFINE_EOS_TOKEN}"


def format_prompt(question):
    return f"###System: {SYSTEM_PROMPT}\n###Question: {question}\n###Answer: "


DEFINE_EOS_TOKEN = '''</s>'''
DEFINE_BOS_TOKEN = '''<s>'''
DEFINE_PAD_TOKEN = '''<pad>'''
SYSTEM_PROMPT = '''You are a robot named "MA-RLHF", you are always friendly and answer questions。'''


# def setup():
#     dist.init_process_group('nccl')

# def cleanup():
#     dist.destroy_process_group()


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


def parse_prompt(text):
    # 正则表达式模式，精确匹配关键字 "system"、"user" 和 "robot"
    pattern = r'\b(###System|###Question|###Answer)\b: (.*?)(?:,\s*|\s*$)'

    # 使用re.findall找到所有匹配的键值对
    matches = re.findall(pattern, text)

    # 将匹配结果转换为字典
    result_dict = {key: value for key, value in matches}

    message = []
    message.append({'role': 'user', 'content': result_dict['###Question']})
    message.append({'role': 'assistant', 'content': result_dict['###Answer']})

    return message

# 示例文本
# text = "system: xxx, user: yyy, robot: zzz"

# # 解析文本并打印结果
# parsed_dict = parse_key_value_pairs(text)
# print(parsed_dict)


def create_dataset(dataset_name, tokenizer):
    data_raw = json.load(dataset_name)
    data_message = [{'chat': parse_prompt(s)} for s in data_raw]
    # load_dataset(dict)
    datasets_hf = datasets.Dataset.from_dict(data_message)
    print(datasets_hf)

    def process(examples):
        prompt = tokenizer.apply_chat_template(examples['prompt'])
        inputs = tokenizer([prompt])
        examples['input_ids'] = inputs['input_ids'][0]
        examples['attention_mask'] = inputs['attention_mask'][0]
        return examples

    datasets = datasets_hf.map(process, num_proc=24)
    datasets = datasets.filter(lambda x: len(x["input_ids"]) < 2048)
    return datasets


def evaluate_reward(model_name, output_name, max_output_tokens, batch_size, dataset_file):
    # evaluation dataset
    # dataset = create_dataset('/root/hh-rlhf') # replace your own dataset

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    if rank == 0:
        if not os.path.exists(output_name):
            os.makedirs(output_name)
            print(f"Folder '{output_name}' created successfully.")
        else:
            print(f"Folder '{output_name}' already exists.")

    dist.barrier()

    model, tokenizer = load_model(model_name, rank)

    dist.barrier()

    dataset = create_dataset(dataset_file, tokenizer)

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

    safe_id = tokenizer.encode('safe')
    unsafe_id = tokenizer.encode('unsafe')

    # time.sleep(100)

    for x in dataloader:
        torch.cuda.empty_cache()
        # len()
        if rank == 0:
            print('-'*50)
            print(f'{step}/{step_all}')
            print('-'*50)
        tmp = {'input_ids': torch.tensor(x['input_ids'], dtype=torch.long).to(rank),
               'attention_mask': torch.tensor(x['attention_mask'], dtype=torch.bool).to(rank)}
        with torch.no_grad():
            y = model(**tmp)['logits'][:, -1, :]

        for i in range(len(x['chat'])):
            tmp_result = {}
            tmp_result['prompt'] = x['chat'][i]['user']
            tmp_result['response'] = x['chat'][i]['assistant']
            tmp_result['safe_prob'] = float(y[i, safe_id])
            tmp_result['unsafe_prob'] = float(y[i, unsafe_id])
            result_all.append(tmp_result)

    current_result_file = output_name + f'/eval_{rank}.json'
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
    parser.add_argument('--dataset_file', type=str, default='', required=True,
                        help='dataset path')
    parser.add_argument('--max_output_tokens', type=int, default=100,
                        help='generation tokens')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='generation tokens')

    args = parser.parse_args()
    print(args)

    evaluate_reward(args.model_name,
                    args.output_name,
                    args.max_output_tokens,
                    args.batch_size
                    args.dataset_file)
