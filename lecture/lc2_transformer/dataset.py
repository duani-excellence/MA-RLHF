import torch
import random
from typing import Dict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
import os


class PaddingCollateFunction:
    def __init__(self, src_pad_token_id: int, trg_pad_token_id: int):
        self.src_pad_token_id = src_pad_token_id
        self.trg_pad_token_id = trg_pad_token_id

    def __call__(self, batch) -> Dict:
        batch = paddding_collate_fn(
            batch, self.src_pad_token_id, self.src_pad_token_id)
        return batch


def paddding_collate_fn(batch_data, src_pad_token_id=None, trg_pad_token_id=None,):

    input_lens = []
    label_lens = []
    bs = len(batch_data)
    for data in batch_data:
        input_lens.append(data['input_ids'].shape[1])
        label_lens.append(data['labels'].shape[1])

    max_input_len = torch.max(torch.tensor(input_lens, dtype=torch.long))
    max_label_len = torch.max(torch.tensor(label_lens, dtype=torch.long))

    input_ids = torch.ones(
        bs, max_input_len, dtype=torch.long) * src_pad_token_id
    input_attention_masks = torch.zeros(bs, max_input_len, dtype=torch.long)
    label_ids = torch.ones(
        bs, max_label_len, dtype=torch.long) * trg_pad_token_id
    label_attention_masks = torch.zeros(bs, max_label_len, dtype=torch.long)

    for i in range(bs):
        input_ids[i, :input_lens[i]
                  ] = batch_data[i]['input_ids'][0, :input_lens[i]]
        input_attention_masks[i, :input_lens[i]] = 1

        label_ids[i, :label_lens[i]
                  ] = batch_data[i]['labels'][0, :label_lens[i]]
        label_attention_masks[i, :label_lens[i]] = 1

    return {
        'input_ids': input_ids,
        'input_attention_mask': input_attention_masks,
        'label_ids': label_ids,
        'label_attention_mask': label_attention_masks,
    }


def load_dataset(filepath):
    """
    文本数据
    """
    if os.path.isfile(filepath):
        with open(filepath, encoding='utf-8') as f:
            data = json.load(f)  # loads 返回 dict
            print('加载成功：')
            return data
    else:
        print(f'[错误] 文件不存在：{filepath}')
        return None


def concat_all_text(data):
    text_zh = ' '.join(data['train']['x']) + ' '.join(data['test']['x'])
    text_en = ' '.join(data['train']['y']) + ' '.join(data['test']['y'])
    # text_all = text_1 + text_2
    return text_en, text_zh


class Seq2SeqTransformersDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):
        return {'input_ids': self.data['input_ids'][idx],
                'labels': self.data['labels'][idx]}


if __name__ == "__main__":
    dataset = load_dataset('./data.json')
    print(dataset['train']['x'][0])
    print(dataset['train']['y'][1])
    text_en, text_zh = concat_all_text(dataset)
    print('en len', len(text_en))
    print('zh len', len(text_zh))