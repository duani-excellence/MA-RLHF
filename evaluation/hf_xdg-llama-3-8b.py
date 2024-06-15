# copy this file to 'opencompass/configs/models/others/hf_xdg-llama-3-8b.py'
# run
# ```
#CUDA_VISIBLE_DEVICES=0 python run.py --models hf_xdg-llama-3-8b --datasets ceval_gen  --num-gpus 1
#```
# also you cloud run : --datasets ceval_gen, mmlu_gen, cmmlu
# result : 42.48

from opencompass.models import HuggingFaceCausalLM

_meta_template = dict(
    begin='###System: You are MA-RLHF Chatbot, you should friendly answer the question\n',
    round=[
        # dict(role='SYSTEM', begin='###SYSTEM: ', end='\n', fallback_role='HUMAN', prompt='Solve the following math questions'),
        dict(role="HUMAN", begin='###Question: ', end='\n'),
        dict(role="BOT", begin="###Answer:",
             generate=True, end='<|end_of_text|>'),
    ],
    # reserved_roles=[dict(role='SYSTEM', begin='###System: ', end='\n', prompt='You are MA-RLHF Chatbot, you should friendly answer the question'), ],
    # eos_token_id=2,
)

models = [
    dict(
        abbr='xdg-llama-3-8b',
        type=HuggingFaceCausalLM,
        path='/mnt/output/llama3-xdg', # you model
        tokenizer_path='/mnt/output/llama3-xdg', # you model
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        meta_template=_meta_template,
        max_out_len=16,
        max_seq_len=1024,
        batch_size=128,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
