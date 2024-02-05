from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
import torch
from utils import ScriptArguments

parser = HfArgumentParser(ScriptArguments)
train_args: ScriptArguments = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]

base_model_name = train_args.base_model_name
model_name = train_args.model_name
merged_model_name = train_args.merged_model_name


def merge(model_base_name, model_adapter_name, model_merge_name):
    # use cpu avoid gpu vram OOM
    # if cpu memory small, use swap
    model = AutoModelForCausalLM.from_pretrained(
        model_base_name, device_map='cpu', torch_dtype=torch.bfloat16, trust_remote_code=True, # llama-7b base
    )
    print(model)

    tokenizer = AutoTokenizer.from_pretrained(
        model_base_name,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(
        model,
        model_adapter_name,  # adapter
        device_map='cpu',
        trust_remote_code=True,
    )
    print(model)

    model = model.merge_and_unload()
    print(model)

    model.save_pretrained(model_merge_name)
    tokenizer.save_pretrained(model_merge_name)


if __name__ == "__main__":
    merge(base_model_name, model_name, merged_model_name)
    print('------merge done!---------')
