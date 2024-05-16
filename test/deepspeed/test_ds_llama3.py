# run scripts
import os
import json
import argparse
import math
import time


from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
from datasets import load_dataset

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import deepspeed
import deepspeed.comm as dist
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam


def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    lora_lr=5e-4,
    no_decay_name_list=[
        "bias", "layer_norm.weight", "layernorm.weight", "norm.weight",
        "ln_f.weight"
    ],
    lora_name_list=["lora_right_weight", "lora_left_weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n.lower()
                                                    for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n.lower()
                                                for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,
            "lr":
            lora_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n.lower()
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]

    non_empty_groups = []
    for group in optimizer_grouped_parameters:
        if group["params"]:
            non_empty_groups.append(group)
    return non_empty_groups

def load_zero3_ckpt():
    '''
        load model shard, not load big model, and then shard
    '''
    return

def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    os.makedirs(save_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)

    model_to_save = model_ema.module if hasattr(model_ema,
                                                'module') else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():

            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v]),
                                                       enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict


def create_config_from_dict(tmpdir, config_dict):
    config_path = os.path.join(tmpdir, 'temp_config.json')
    with open(config_path, 'w') as fd:
        json.dump(config_dict, fd)
    return config_path


def get_data_loader(model, total_samples, hidden_dim, device):
    batch_size = model.train_micro_batch_size_per_gpu()
    train_data = torch.randn(total_samples, hidden_dim, device=device, dtype=torch.half)
    train_label = torch.empty(total_samples, dtype=torch.long, device=device).random_(hidden_dim)
    print(train_data.shape)
    print(train_label.shape)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler
    )
    return train_loader


def get_args(config_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--zero', type=int, default=0)
    args = parser.parse_args()  # args=''

    config_path = os.path.join(tmpdir, 'temp_config.json')
    with open(config_path, 'w') as fd:
        json.dump(config_dict, fd)


    config_dict["zero_optimization"]["stage"] = args.zero
    print('config_dict["zero_optimization"]', config_dict["zero_optimization"])
    args.deepspeed_config = config_path
    return args


def print0(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output

def load_model_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def process_fn(example, tokenizer):
    inst = example['instruction']
    input = example['input']
    output = example['output']
    prompt = f'###USER:{inst} {input}\n###ASSISTANT:{output}'
    prompt_id = tokenizer.encode(prompt, return_tensor='pt', max_length=512, trunction=True)
    example['input_ids'] = prompt_id['input_ids']
    example['attention_masks'] = prompt_id['attention_masks']
    example['labels'] = prompt_id['input_ids'].clone()
    return example


def load_dataset_ddp(name, tokenizer):
    datasets = load_dataset(name)
    datasets.map(process_fn, tokenizer, batched=False, remove_columns=['instruction', 'input', 'output'])
    return datasets

def get_dataloader(datasets, args):
    train_sampler = DistributedSampler(datasets)
    train_dataloader = DataLoader(
        datasets,
        collate_fn=DataCollatorWithPadding,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
    )
    return train_dataloader


def get_optimizer(model, args):
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay, args.lora_learning_rate)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            betas=(0.9, 0.95))
    return optimizer


def get_scheduler(optimizer, data_loader, args):
    num_update_steps_per_epoch = math.ceil(len(data_loader) / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type, # cosine
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )
    return lr_scheduler


def train(model, train_dataloader, args):
    for epoch in range(args.num_train_epochs):
        model.train()

        for step, batch in enumerate(train_dataloader):
            start = time.time()
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            model.backward(loss)
            model.step()
            end = time.time()
            if torch.distributed.get_rank() == 0:
                print(
                    f"Epoch: {epoch}, Step: {step}, Rank: {torch.distributed.get_rank()}, loss = {loss}"
                )
                print(
                    f"time {end-start} it/s"
                )

        save_zero_three_model(
            model, args.global_rank, args.output_dir, zero_stage=args.zero_stage
        )


def run(args):
    rank = int(os.environ['RANK'])
    torch.random.manual_seed(42)
    torch.distributed.barrier()

    model, tokenizer = load_model_tokenizer('xiaodongguaAIGC/xdg-llama-3-8B')
    datasets = load_dataset_ddp('xiaodongguaAIGC/alpaca_en_zh_ruozhiba', tokenizer)
    data_loader = get_dataloader(datasets['train'], args)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_scheduler(optimizer, data_loader, args)

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=args,
        lr_scheduler=lr_scheduler,
        dist_init_required=True,
    )
    model.gradient_checkpointing_enable()

    # training
    train(model, optimizer, args)


if __name__ == "__main__":
    args = get_args('./config/ds_config_llama.json')
    args.output_dir = './output/deepspeed_llama'
    args.lr_scheduler_type = 'consine'
    args.learning_rate = 1e-5
    args.num_warmup_steps = 10
    args.num_train_epochs = 2
    args.weight_decay = 0.
    args.offload = None
    args.global_rank = 0
    run(args)
