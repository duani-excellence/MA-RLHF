# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./evaluation/vllm_generation.py \
# 	--model_name='{YOU MODEL}' \
# 	--output_name='./output/result_vllm_tmp.json'  \
# 	--max_output_tokens=512 \
# 	--num_gpus=4


import re
import numpy as np
from datasets import load_dataset, load_from_disk
from vllm import LLM, SamplingParams
import argparse
import json


# to do use utils
SYSTEM_PROMPT = '''You are a robot named "MA-RLHF", you are always friendly and answer questions。'''

def format_prompt(question):
    return f"###System: {SYSTEM_PROMPT}\n###Question: {question}\n###Answer: "


def create_dataset(dataset_name):
    datasets = load_dataset(dataset_name, split='test')
    def preprocess_function(examples):
        new_examples = {
            "prompts": [],
        }
        for question in examples["prompt"]:
            query = format_prompt(question)
            new_examples["prompts"].append(query)
        return new_examples

    def preprocess_function_hhrlhf(examples):
        new_examples = {
            "prompts": [],
        }
        for prompt_chosen in examples["chosen"]:
            # format hh-rlhf dataset for PPO
            prompt_chosen = prompt_chosen.rsplit('Assistant:',1)[0]
            prompt_chosen = re.sub(r'Human:', '###Question:', prompt_chosen)
            prompt_chosen = re.sub(r'Assistant:', '###Answer:', prompt_chosen)
            # query = prompt_chosen + '###Answer:'
            prompt_question = prompt_chosen.split('\n###Answer:', 1)[0] + '\n###Answer:'
            new_examples['prompts'].append(prompt_question)

        return new_examples

    func = None
    # if dataset_name == '/root/hh-rlhf':
    #     func = preprocess_function_hhrlhf
    # elif dataset_name == '/root/PKU-SafeRLHF-30K':
    func = preprocess_function

    datasets = datasets.map(
        func,
        batched=True,
        num_proc=8,
        # remove_columns=datasets.column_names,
    )

    datasets = datasets.filter(lambda x: len(x["prompts"]) < 512)
    # datasets.set_format(type="torch")
    return datasets

def vllm_generate(model_name, output_name, max_output_tokens, batch_size, num_gpus):
    # evaluation dataset
    # dataset = create_dataset('/root/hh-rlhf') # replace your own dataset
    dataset = create_dataset('PKU-Alignment/PKU-SafeRLHF-30K')
    print(dataset)

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8,
                                     max_tokens=max_output_tokens,
                                     stop="</s>",
                                    #  use_cache=False,
                                     )
    # Create LLM object
    # kwargs
    llm = LLM(model=model_name, # replace your own model
              dtype='bfloat16',
              tensor_parallel_size=num_gpus,  # number of gpu
              gpu_memory_utilization=0.9,  # prevent OOM
            #   use_cache=False,
              )

    # # load toxicity evaluation model
    # toxicity = evaluate.load("./metrics/toxicity", module_type="measurement") # add toxicity script to evaluate package

    # # vllm generation
    outputs = llm.generate(dataset['prompts'],
                           sampling_params,)


    result_all = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        # print('-'*50)
        # print(prompt)
        # print(generated_text)
        tmp = {'prompt':prompt,
                'response':generated_text}
        result_all.append(tmp)


    current_result_file = output_name
    with open(current_result_file, 'w') as file:
        json.dump(result_all, file)




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
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='generation tokens')

    args = parser.parse_args()
    print(args)

    vllm_generate(args.model_name,
                args.output_name,
                args.max_output_tokens,
                args.batch_size,
                args.num_gpus)
    # print(f'toxicity score mean: {mean}, toxicity score std: {std}')
