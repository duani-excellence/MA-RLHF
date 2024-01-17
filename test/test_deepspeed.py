from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# get dataset
dataset = load_dataset("imdb", split="train")


training_args = TrainingArguments(
    output_dir='./output/output_test_deepspeed',
    deepspeed='./config/ds.json',
)

# get trainer
trainer = SFTTrainer(
    "gpt2",
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
)

# train
trainer.train()
