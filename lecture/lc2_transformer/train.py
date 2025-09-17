# python train.py --learning_rate 1e-4 --epochs 1

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from tokenizer import TokenizerBase
from model import Transformer
from dataset import load_dataset, Seq2SeqTransformersDataset, PaddingCollateFunction
from config import TransformerModelConfig
from utils import (
    PAD_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    IGNORE_INDEX,
    token_pre_process,
    get_argparse
)

torch.manual_seed(42)


def process(data,
            batch_size: int = 2,
            src_tokenizer: TokenizerBase = None,
            trg_tokenizer: TokenizerBase = None):
    print(len(data['x']))

    _, encode_x = src_tokenizer.encode(data['x'])
    _, encode_y = trg_tokenizer.encode(data['y'])

    # padding and truction
    encode_x = token_pre_process(encode_x,
                                 sos_token_id=src_tokenizer.vocab[SOS_TOKEN],
                                 eos_token_id=src_tokenizer.vocab[EOS_TOKEN])
    encode_y = token_pre_process(encode_y,
                                 sos_token_id=trg_tokenizer.vocab[SOS_TOKEN],
                                 eos_token_id=trg_tokenizer.vocab[EOS_TOKEN]
                                 )
    tensor_list_x = [torch.tensor([x]) for x in encode_x]
    tensor_list_y = [torch.tensor([y]) for y in encode_y]

    dataset = Seq2SeqTransformersDataset(
        {'input_ids': tensor_list_x, 'labels': tensor_list_y})
    # datacollate
    collate_fn = PaddingCollateFunction(src_pad_token_id=src_tokenizer.vocab[EOS_TOKEN],
                                        trg_pad_token_id=trg_tokenizer.vocab[EOS_TOKEN])

    # dataloader
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            collate_fn=collate_fn,
                            pin_memory=True,
                            shuffle=True)

    return dataloader


if __name__ == "__main__":

    parser = get_argparse()
    args = parser.parse_args()

    # load tokenizer
    src_tokenizer = TokenizerBase()
    src_tokenizer.from_pretrained(args.src_tokenizer_path)
    trg_tokenizer = TokenizerBase()
    trg_tokenizer.from_pretrained(args.trg_tokenizer_path)

    # create model
    config = TransformerModelConfig(
        src_vocab_size=src_tokenizer.config.vocab_size,
        trg_vocab_size=trg_tokenizer.config.vocab_size,
        max_len=512,
        dim=16,
        heads=8,
        num_layers=1,
        position_encoding_base=10000.0,
        src_pad_token_id=src_tokenizer.vocab[PAD_TOKEN],
        trg_pad_token_id=trg_tokenizer.vocab[PAD_TOKEN],
    )
    model = Transformer(config)

    # load dataset
    data = load_dataset('data.json')
    train_dataloader = process(
        data['train'], src_tokenizer=src_tokenizer, trg_tokenizer=trg_tokenizer, batch_size=args.batch_size)
    test_dataloader = process(
        data['test'], src_tokenizer=src_tokenizer, trg_tokenizer=trg_tokenizer, batch_size=args.batch_size)

    # train
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    epochs = args.epochs
    train_loss = []
    test_loss = []
    total_step = 0

    # training
    for i in range(epochs):

        train_dataloader_tqdm = tqdm(
            train_dataloader,  # 数据加载器
            desc=f'Epoch {i+1}/{epochs}',  # 进度条前缀
            ncols=100,    # 进度条宽度
            ascii=' =',   # ASCII 字符样式
            leave=False   # 完成后不保留进度条
        )

        # for batch in train_dataloader:
        for batch in train_dataloader_tqdm:
            optimizer.zero_grad()
            X = batch['input_ids']
            Y = batch['label_ids'][:, :-1]

            logits, _, _, _ = model(X, Y, )

            label = batch['label_ids'].clone()[:, 1:]
            bs, tmp_trg_len = label.shape
            label[torch.where(label == trg_tokenizer.vocab[PAD_TOKEN])
                  ] = IGNORE_INDEX
            loss = loss_fn(logits.reshape(bs*tmp_trg_len, trg_tokenizer.config.vocab_size),
                           label.reshape(bs * tmp_trg_len))

            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

            total_step = total_step + 1

            if total_step % 10 == 0:
                # print(
                # f"epochs:{i}, step:{total_step}, train_loss: {loss.item()}")
                # tqdm.write("\n" + "=" * 80)
                tqdm.write(f"Epoch {i+1} | "
                           f"Steps {total_step} | "
                           f"Loss: {loss.item():.4f} | ")

            train_dataloader_tqdm.set_postfix(
                loss=f'{loss.item():.4f}',
            )

    # evaluation

    # save_pretrained
    model.save_pretrained(args.output_path)
    src_tokenizer.save_pretrained(args.output_path + '/tokenizer_zh')
    trg_tokenizer.save_pretrained(args.output_path + '/tokenizer_en')
