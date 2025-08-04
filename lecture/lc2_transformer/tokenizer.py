from typing import List, Dict, Tuple, Union, str, Any
from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass, asdict
import re
import string
from dataclasses import asdict
import json
import os

def save_dict_to_json(filepath, data):
    """将字典保存为 JSON 文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=True, indent=4)
    print(f"字典已保存为 JSON 文件: {filepath}")




class SpecialToken:
    def __init__(self,):
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
special_token = SpecialToken()


@dataclass
class TokenizerBaseConfig:
    vocab_size: int = -1
    class_name: str = 'TokenizerBase'
    sos_token: str = '<SOS>'
    sos_token_id: int = -1
    eos_token: str = '<EOS>'
    eos_token_id: int = -1
    pad_token: str = '<PAD>'
    pad_token_id: int = -1
    unk_token: str = '<UNK>'
    unk_token_id: int = -1
    pattern: str = ''


    



class TokenizerBase():
    # @abstractmethod
    def __init__(self, config : TokenizerBaseConfig= None):
        self.vocab: Dict[str, int] = {}
        self.vocab_reverse: Dict[int, str] = {}
        self.vocab_size: int = 0
        self.special_token: Dict[str, int] = {}

        self.config = config


        special_tokens = ['<SOS>', '<EOS>', '<PAD>', '<UNK>']
        zh_symbols = '，。！？；：“”‘’【】（）《》、'
        en_symbols = re.escape(string.punctuation)
        all_symbols = zh_symbols + en_symbols + ' '
        self.pattern = (
            r'(?:' + '|'.join(special_tokens) + ')'   
            r'|[' + re.escape(all_symbols) + ']' 
            r'|\d'  
            r'|[\u4e00-\u9fa5]'  
            r'|[^' + re.escape(all_symbols) + r'\d\u4e00-\u9fa5<>]+'  
        )

    # @abstractmethod
    def init_vocab(self, vocab: Dict[str, int]):
        """
        初始化词表, 可以用现成的字典, 也可以默认使用基础字符来创建
        """

    # @abstractmethod
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
        token_init_list = re.findall(self.pattern, text_init)
        token_corpus_list = re.findall(self.pattern, text)

        token_all = token_init_list + token_corpus_list

        # vocab : Dict[str, int] = {}
        # vocab_reverse: Dict[str, int] = {}
        idx = 0
        for value in token_all:
            if value not in self.vocab:
                self.vocab[value] = idx
                self.vocab_reverse[idx] = value
                idx += 1
        

    # @abstractmethod
    def add_special_token(self, token: Dict[str, str]):
        """
        添加特殊 token, 存入 特殊的 tokenizer 表中
        """
        pass

    # @abstractmethod
    def encode(self,input_list : List[str] =[],
               padding : bool = True,
               padding_side : str = "left",
               max_length : Union[int, str] ='right',
               add_bos_token : bool = False,
               add_eos_token : bool = False,
               add_pad_token : bool = False,
               return_type : str = str,  # pt: pytorch tensor
               ):
        
        token_list = []
        token_ids_list = []
        for ids in input_list:
            tokens = re.findall(self.pattern, input_list) # 分词规则
            token_ids = []
            for token in tokens:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else: 
                    if len(token) == 1:
                        token_ids.append(self.vocab['<UNK>'])
                    else:
                        for t in token:
                            token_ids.append( self.vocab[t] )
            token_list.append(token)
            token_ids_list.append(token_ids_list)
        return token_list, token_ids_list

    @abstractmethod
    def decode(self, token_ids: list[list[int]],
               skip_special_token : bool =True,
               return_string : bool=True
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
    def from_pretrained(self, filepath : str ='./tokenizer'):
        pass

    @abstractmethod
    def save_tokenizer(self, filepath : str ='./tokenizer'):
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

        vocab_path = os.path.join(filepath, 'vocab.json')
        config_path = os.path.join(filepath, 'config.json')

        config_dict = asdict(self.config)

        save_dict_to_json(config_path, config_dict)
        save_dict_to_json(vocab_path, self.vocab)        

    @abstractmethod
    def chat_template(self,
                      prompt : Union[str, List[str]] =None,
                      response : Union[str, List[str]] =None,
                      messages :  List[Dict[str, Any]]  =None,
                      tokenize:bool =  False,
                      add_response_prompt : bool =False,):
        pass
