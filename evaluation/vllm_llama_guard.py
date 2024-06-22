# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./evaluation/vllm_generation.py \
# 	--model_name='meta-llama/Meta-Llama-Guard-2-8B' \
#   --dataset_name='./output/result_vllm_generation' \
# 	--output_name='./output/result_vllm_llamaguard'  \
# 	--max_output_tokens=1 \
# 	--num_gpus=4

import re
import numpy as np
from datasets import load_dataset, load_from_disk, DatasetDict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse
import json


def create_dataset(dataset_name, tokenizer):
    # datasets = load_dataset(dataset_name, split='test')
    datasets = load_from_disk(dataset_name)['test']

    # chat messages
    prompts = []
    for dataset in datasets:
        # print(dataset)
        chat_messages = [{'role':'user', 'content': dataset['prompt']},
                         {'role':'assistant', 'content': dataset['response']}]
        # print(chat_messages)
        llama_guard_prompt = tokenizer.apply_chat_template(chat_messages, tokenize=False)
        prompts.append(llama_guard_prompt)
    return datasets, prompts


def vllm_generate(model_name, dataset_name ,output_name, max_output_tokens, batch_size, num_gpus):
    # evaluation dataset
    # dataset = create_dataset('/root/hh-rlhf') # replace your own dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    SAFE_TOKEN_ID, UNSAFE_TOKEN_ID = [token_id for *_, token_id in tokenizer(['safe', 'unsafe'])['input_ids']]


    dataset, llama_guard_prompt = create_dataset(dataset_name, tokenizer)
    # dataset = prepare_dataset(dataset, tokenizer)


    print(llama_guard_prompt[:5])

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8,
                                     max_tokens=max_output_tokens, # must be 1
                                    #  stop="</s>",
                                    #  max_tokens = 1,
                                     top_k = 10,
                                    top_p = 0.95,
                                    # temperature = 0,
                                    frequency_penalty = 0,
                                    # max_tokens = 1,
                                    logprobs = 5,
                                     )
    # Create LLM object
    # kwargs
    llm = LLM(model=model_name, # replace your own model
              dtype='bfloat16',
              tensor_parallel_size=num_gpus,  # number of gpu
              gpu_memory_utilization=0.95,  # prevent OOM
              max_num_seqs = batch_size,
              )


    # # vllm generation
    outputs = llm.generate(llama_guard_prompt,
                           sampling_params,)


    # result_all = []
    answers = []
    for output in outputs:
        final_answer = output.outputs[0].logprobs[0]
        if SAFE_TOKEN_ID not in final_answer:
            safe_prob = np.exp(float('-inf'))
        else:
            safe_prob = np.exp(final_answer[SAFE_TOKEN_ID].logprob)
        answers.append(safe_prob)

    print(answers)

    current_result_file = output_name + '.json'
    with open(current_result_file, 'w', encoding='utf-8') as file:
        json.dump(answers, file, )

    # save hf dataset
    dataset.add_column(name='safe_prob', column=answers)
    dataset_dict = DatasetDict({'test':dataset})
    dataset_dict.save_to_disk(output_name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_name',  type=str, default='meta-llama/Meta-Llama-Guard-2-8B',
                        help='model name path')
    parser.add_argument('--dataset_name',  type=str, default='', required=True,
                        help='dataset name path')
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
                    args.dataset_name,
                    args.output_name,
                    args.max_output_tokens,
                    args.batch_size,
                    args.num_gpus)
    # print(f'toxicity score mean: {mean}, toxicity score std: {std}')
