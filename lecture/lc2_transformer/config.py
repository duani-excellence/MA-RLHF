from dataclasses import dataclass, asdict, field
import json


@dataclass
class TransformerModelConfig:
    learning_rate: float = 0.001
    src_vocab_size: int = 100
    trg_vocab_size: int = 200
    max_len: int = 512
    dim: int = 512
    heads: int = 8
    num_layers: int = 6
    position_encoding_base: float = 10000.0
    src_pad_token_id: int = 0
    trg_pad_token_id: int = 0
    # custom_objects: dict = field(default_factory=dict)

    def save_json(self, path):
        config_dict = asdict(self)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
