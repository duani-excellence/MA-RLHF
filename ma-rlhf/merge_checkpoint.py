from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
import torch
from utils import ScriptArguments, create_peft, create_peft_reward_model
from peft import get_peft_model_state_dict, PeftModel, LoraConfig
import peft
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

parser = HfArgumentParser(ScriptArguments)
train_args: ScriptArguments = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]

base_model_name = train_args.base_model_name
model_checkpoint_name = train_args.model_name
model_adapter_name = train_args.merged_model_name
merge_checkpoint_type = train_args.merge_checkpoint_type

def merge(model_base_name, model_checkpoint_name, model_adapter_name):
    # use cpu avoid gpu vram OOM
    # if cpu memory small, use swap
    model = AutoModelForCausalLM.from_pretrained(
        model_base_name, device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True, # llama-7b base
    )
    print('load base model')

    tokenizer = AutoTokenizer.from_pretrained(
        model_base_name,
        trust_remote_code=True,
    )

    peft_config = None
    if merge_checkpoint_type == 'LM':
        peft_config = create_peft(True)
    elif merge_checkpoint_type == 'Reward':
        peft_config = create_peft_reward_model(True)

    model = peft.PeftModel(model, peft_config)
    state_dict = get_fp32_state_dict_from_zero_checkpoint(model_checkpoint_name) # already on cpu
    d = get_peft_model_state_dict(model, state_dict=state_dict)
    model.save_pretrained(model_adapter_name)
    tokenizer.save_pretrained(model_adapter_name)
    print('save model finish')


if __name__ == "__main__":
    merge(base_model_name, model_checkpoint_name, model_adapter_name)
    print('------merge done!---------')
