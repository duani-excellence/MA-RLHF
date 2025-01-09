from typing import Dict, List, Optional, Union, Any
import torch
import transformers

from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
)

from utils import (
    format_prompt,
    DEFINE_SEP_TOKEN,
    STEP_INSTRUCTION,
    PRM_INSTRUCTION,
)


# STEP_INSTRUCTION = '''Solve this math problem using step-by-step reasoning.\n'''
def process_sft_step(example, tokenizer = None):
    question = example['prompt']
    steps = example['completions']

    # 分词prompt
    prompt_format = format_prompt(STEP_INSTRUCTION + question)
    prompt_encode = tokenizer.encode(prompt_format, return_tensors='pt')[0]
    prompt_len = prompt_encode.shape[0]

    # 分词response
    response = ''
    for step in steps:
        response = response + step + DEFINE_SEP_TOKEN
    response = response + tokenizer.eos_token
    response_encode = tokenizer.encode(response, return_tensors='pt')[0]
    response_encode = response_encode[1:] # ignore <bos> token

    # 拼接 prompt + response
    input_ids = torch.cat((prompt_encode, response_encode), dim=0)
    attention_mask = torch.ones_like(input_ids)

    # 创建 label
    prompt_label = torch.clone(input_ids)
    prompt_label[:prompt_len] = -100  # 非response ignore
    # prompt_label[-1] = -100  # 非response ignore

    # 不用手动shift， llama model计算loss时会自动shift
    # prompt_label = torch.roll(prompt_label, shifts=-1)  # label左移动一位，形成casual mask

    example['input_ids'] = input_ids
    example['attention_mask'] = attention_mask
    example['label_ids'] = prompt_label
    # example['labels'] = prompt_label
    # example['len']=prompt_len

    return example


# positive_id = prm_tokenizer('Positive', add_special_tokens=False)['input_ids']
# positive_token = 'Positive'
# print(positive_id)

# negative_id = prm_tokenizer('Negative', add_special_tokens=False)['input_ids']
# negative_token = 'Negative'
# print(negative_id)
label_map = {0: 36590, 1: 39589}


def process_prm_step(example,  tokenizer = None):
    question = example['prompt']
    steps = example['completions']
    labels = example['labels']

    # 分词prompt
    prompt_format = format_prompt(PRM_INSTRUCTION + question)
    prompt_encode = tokenizer.encode(prompt_format, return_tensors='pt')[0]
    prompt_len = prompt_encode.shape[0]

    response_token_ids = []
    place_indexs = []
    label_idx = []

    for step, label in zip(steps, labels):
        response = step + DEFINE_SEP_TOKEN  # step的尾部加上sep token
        step_response_token_ids = tokenizer.encode(
            response, add_special_tokens=False)

        response_token_ids.extend(step_response_token_ids)
        place_indexs.append(len(response_token_ids) + prompt_len)
        label_idx.append(label_map[label])

    response_token_ids.extend([tokenizer.eos_token_id])  # 完整结束后增加eos token
    response_token_ids = torch.tensor(response_token_ids)

    # 拼接 prompt + response
    input_ids = torch.cat((prompt_encode, response_token_ids), dim=0)
    attention_mask = torch.ones_like(input_ids)
    # attention_mask[place_indexs] = False

    # 创建 label， 在sep token里才有回归的标签
    # 这里相当于一长串数据，可以一次性回归多个sep_token 对应的correctness的标签
    place_indexs = [idx for idx in place_indexs] # 同step sft
    prompt_label = torch.ones_like(input_ids) * -100
    prompt_label[place_indexs] = torch.tensor(label_idx, dtype=torch.long)

    print(input_ids)
    print(prompt_label)

    example['input_ids'] = input_ids
    example['attention_mask'] = attention_mask
    example['label_ids'] = prompt_label

    return example



def process_instruction(example,  tokenizer = None):
    question = example['instruction'] + example['input']
    steps = [example['output']]
    labels = [True]

    # 分词prompt
    prompt_format = format_prompt(PRM_INSTRUCTION + question)
    prompt_encode = tokenizer.encode(prompt_format, return_tensors='pt')[0]
    prompt_len = prompt_encode.shape[0]

    response_token_ids = []
    place_indexs = []
    label_idx = []

    for step, label in zip(steps, labels):
        response = step + DEFINE_SEP_TOKEN  # step的尾部加上sep token
        step_response_token_ids = tokenizer.encode(
            response, add_special_tokens=False)

        response_token_ids.extend(step_response_token_ids)
        place_indexs.append(len(response_token_ids) + prompt_len)
        label_idx.append(label_map[label])

    response_token_ids.extend([tokenizer.eos_token_id])  # 完整结束后增加eos token
    response_token_ids = torch.tensor(response_token_ids)

    # 拼接 prompt + response
    input_ids = torch.cat((prompt_encode, response_token_ids), dim=0)
    attention_mask = torch.ones_like(input_ids)
    # attention_mask[place_indexs] = False

    # 创建 label， 在sep token里才有回归的标签
    # 这里相当于一长串数据，可以一次性回归多个sep_token 对应的correctness的标签
    place_indexs = [idx - 1 for idx in place_indexs]
    prompt_label = torch.ones_like(input_ids) * -100
    prompt_label[place_indexs] = torch.tensor(label_idx, dtype=torch.long)

    example['input_ids'] = input_ids
    example['attention_mask'] = attention_mask
    example['label_ids'] = prompt_label

    return example


def process_orm_step(example,  tokenizer = None):
    question = example['prompt']
    steps = example['completions']
    labels = example['labels']

    # 分词prompt
    prompt_format = format_prompt(question)
    prompt_encode = tokenizer.encode(prompt_format, return_tensors='pt')[0]
    prompt_len = prompt_encode.shape[0]

    response_token_ids = []
    place_indexs = []
    label_idx = []

    for step, label in zip(steps, labels):
        response = step + DEFINE_SEP_TOKEN  # step的尾部加上sep token
        step_response_token_ids = tokenizer.encode(
            response, add_special_tokens=False)

        response_token_ids.extend(step_response_token_ids)
        place_indexs.append(len(response_token_ids) + prompt_len)
        label_idx.append(label_map[label])

    response_token_ids.extend([tokenizer.eos_token_id])  # 完整结束后增加eos token
    response_token_ids = torch.tensor(response_token_ids)

    # 拼接 prompt + response
    input_ids = torch.cat((prompt_encode, response_token_ids), dim=0)
    attention_mask = torch.ones_like(input_ids)

    # 创建 label， 在sep token里才有回归的标签
    place_indexs = place_indexs[-1] -1 # 仅最后一个sep token 才做回归
    prompt_label = torch.ones_like(input_ids) * -100
    prompt_label[place_indexs] = torch.tensor(label_idx, dtype=torch.long)

    example['input_ids'] = input_ids
    example['attention_mask'] = attention_mask
    example['label_ids'] = prompt_label

    return example


class DataCollatorForSFT(DataCollatorWithPadding):
    """
    继承DataCollatorWithPadding实现动态padding
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str,
                   transformers.tokenization_utils_base.PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    # features: List[Dict[str, Union[List[int], torch.Tensor]]]

    # def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 分离input和label
        input_ids = [{"input_ids": f["input_ids"]} for f in features]

        # 动态padding input
        batch = self.tokenizer.pad(
            input_ids,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        labels_list = [f["label_ids"] for f in features]
        max_len = max([len(labels) for labels in labels_list])
        # print(max_len)
        target_labels = []

        for labels in labels_list:
            labels = [-100] * (max_len - len(labels)) + labels
            target_labels.append(labels)
        batch["labels"] = torch.tensor(target_labels)

        return batch
