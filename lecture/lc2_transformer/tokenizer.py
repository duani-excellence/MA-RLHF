from abc import abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Dict, Union
import re
import string
import json
import os

from dataset import load_dataset, concat_all_text
from utils import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, save_dict_to_json


@dataclass
class TokenizerBaseConfig:
    vocab_size: int = -1
    class_name: str = 'TokenizerBase'
    sos_token: str = SOS_TOKEN
    sos_token_id: int = -1
    eos_token: str = EOS_TOKEN
    eos_token_id: int = -1
    pad_token: str = PAD_TOKEN
    pad_token_id: int = -1
    unk_token: str = UNK_TOKEN
    unk_token_id: int = -1
    pattern: str = ''


class TokenizerBase:
    # @abstractmethod
    def __init__(self, config: TokenizerBaseConfig = None):
        self.vocab: Dict[str, int] = {}
        self.vocab_reverse: Dict[int, str] = {}
        self.config = config

        if config is None:
            self.config = TokenizerBaseConfig()
            special_tokens = [self.config.sos_token,
                              self.config.eos_token,
                              self.config.pad_token,
                              self.config.unk_token,]
            zh_symbols = '，。！？；：“”‘’【】（）《》、'
            en_symbols = re.escape(string.punctuation)
            all_symbols = zh_symbols + en_symbols + ' '

            self.config.pattern = (
                r'(?:' + '|'.join(special_tokens) + ')'
                r'|[' + re.escape(all_symbols) + ']'
                r'|\d'
                r'|[\u4e00-\u9fa5]'
                r'|[^' + re.escape(all_symbols) + r'\d\u4e00-\u9fa5<>]+'
            )

    def train(self, text: Union[str, List[str]]):
        """
        输入语料
        """
        text_init = """ 
         a b c d e f g h i j k l m n o p q r s t u v w x y z 
         A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 
         0 1 2 3 4 5 6 7 8 9 10 
         <SOS> <EOS> <UNK> <PAD> 
         , 。 ！？；：“”‘’【】（）《》、!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~ 
        """
        token_init_list = re.findall(self.config.pattern, text_init)
        token_corpus_list = re.findall(self.config.pattern, text)

        token_all = token_init_list + token_corpus_list

        idx = 0
        for value in token_all:
            if value not in self.vocab:
                self.vocab[value] = idx
                self.vocab_reverse[idx] = value
                idx += 1
        self.config.vocab_size = len(self.vocab)

    # @abstractmethod

    def add_special_token(self, token: Dict[str, str]):
        """
        添加特殊 token, 存入 特殊的 tokenizer 表中
        """
        pass

    # @abstractmethod
    def encode(self, input_list: List[str] = [],
               padding: bool = True,
               padding_side: str = "left",
               max_length: Union[int, str] = 'right',
               add_bos_token: bool = False,
               add_eos_token: bool = False,
               add_pad_token: bool = False,
               return_type: str = str,  # pt: pytorch tensor
               ):

        token_list = []
        token_ids_list = []
        for input_text in input_list:
            tokens = re.findall(self.config.pattern, input_text)  # 分词规则
            token_ids = []
            for token in tokens:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    for t in token:
                        if t not in self.vocab:
                            token_ids.append(self.vocab[UNK_TOKEN])
                        else:
                            token_ids.append(self.vocab[t])
            token_list.append(tokens)
            token_ids_list.append(token_ids)
        return token_list, token_ids_list

    @abstractmethod
    def decode(self, token_ids: List[List[int]],
               skip_special_token: bool = True,
               return_string: bool = True
               ):
        """
        批量解码
        """
        decode_token_list = []
        for ids in token_ids:
            decode_token = []
            for id in ids:
                decode_token.append(self.vocab_reverse[id])
            decode_token_list.append(decode_token)
        return decode_token_list

    @abstractmethod
    def from_pretrained(self, filepath: str = './tokenizer'):
        vocab_path = os.path.join(filepath, 'vocab.json')
        config_path = os.path.join(filepath, 'config.json')

        if os.path.isfile(config_path):
            with open(config_path, encoding='utf-8') as f:
                config = json.load(f)  # loads 返回 dict
                # print('加载成功：')
        else:
            print(f'[错误] 文件不存在：{config_path}')

        if 'class_name' in config:
            cls = globals()[config['class_name']]  # 获取类对象
            self.config = cls(**config)
        else:
            print('not specified tokenizer class name')

        if os.path.isfile(vocab_path):
            with open(vocab_path, encoding='utf-8') as f:
                self.vocab = json.load(f)  # loads 返回 dict
                for value in self.vocab:
                    self.vocab_reverse[self.vocab[value]] = value
        else:
            print(f'[错误] 文件不存在：{vocab_path}')

        return

    @abstractmethod
    def save_pretrained(self, filepath: str = './tokenizer'):
        """
        保存 tokenizer, 包含词表, 分词规则, config
        config 保存 分词器 类名, 分词器保存规则 
        """
        if not os.path.exists(filepath):
            os.makedirs(filepath)
            print(f"目录 '{filepath}' 已创建")
        else:
            print(f"目录 '{filepath}' 已存在")
            # return False

        config_dict = {
            'vocab_size': len(self.vocab),
            'class_name': 'TokenizerBaseConfig',
            'sos_token': SOS_TOKEN,
            'sos_token_id': self.vocab[SOS_TOKEN],
            'eos_token': EOS_TOKEN,
            'eos_token_id': self.vocab[EOS_TOKEN],
            'pad_token': PAD_TOKEN,
            'pad_token_id': self.vocab[PAD_TOKEN],
            'unk_token': UNK_TOKEN,
            'unk_token_id': self.vocab[UNK_TOKEN],
            'pattern': self.config.pattern,
        }
        self.config = TokenizerBaseConfig(**config_dict)
        config_dict = asdict(self.config)

        vocab_path = os.path.join(filepath, 'vocab.json')
        config_path = os.path.join(filepath, 'config.json')
        save_dict_to_json(config_path, config_dict)
        save_dict_to_json(vocab_path, self.vocab)

    # @abstractmethod
    # def chat_template(self,
    #                   prompt : Union[str, List[str]] =None,
    #                   response : Union[str, List[str]] =None,
    #                   messages :  List[Dict[str, Any]]  =None,
    #                   tokenize:bool =  False,
    #                   add_response_prompt : bool =False,):
    #     pass


def test():
    # init
    print('-'*100)
    print('[init]....')
    tokenizer = TokenizerBase()

    # train
    print('-'*100)
    print('[training]....')
    tokenizer.train('')

    # save
    print('-'*100)
    print('[saving]....')
    tokenizer.save_pretrained('./output')
    del tokenizer

    # load
    print('-'*100)
    print('[loading]....')
    tokenizer = TokenizerBase()
    tokenizer.from_pretrained('./output')
    print(tokenizer.config.vocab_size)
    print(tokenizer.config)

    # encode
    print('-'*100)
    print('[encoding]....')
    texts = ['I love Xiao Dong Gua AIGC', 'I have a dream']
    token_list, token_ids_list = tokenizer.encode(texts)
    for text, token, token_ids in zip(texts, token_list, token_ids_list):
        print(text)
        print(token)
        print(token_ids)

    # decode
    print('-'*100)
    print('[decoding]....')
    decode_texts = tokenizer.decode(token_ids_list)
    for decode_text in decode_texts:
        print('-'*100)
        print(decode_text)
        print(''.join(decode_text))


if __name__ == "__main__":
    # test
    test()

    # train tokenizer for Transformer model
    dataset = load_dataset('./data.json')
    text_en, text_zh = concat_all_text(dataset)

    tokenizer_en = TokenizerBase()
    tokenizer_en.train(text_en)
    tokenizer_en.save_pretrained('./output/tokenizer_en')
    print('English Vocab Size:', tokenizer_en.config.vocab_size)

    tokenizer_zh = TokenizerBase()
    tokenizer_zh.train(text_zh)
    tokenizer_zh.save_pretrained('./output/tokenizer_zh')
    print('Chinese Vocab Size:', tokenizer_zh.config.vocab_size)
