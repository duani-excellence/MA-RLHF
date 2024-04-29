
from datasets import load_dataset, concatenate_datasets, DatasetDict
import pprint

data_name1 = 'silk-road/alpaca-data-gpt4-chinese'
data_name2 = 'vicgalle/alpaca-gpt4'
data_name3 = 'LooksJuicy/ruozhiba'


print('load 2 dataset')
dataset1 = load_dataset(data_name1)
dataset2 = load_dataset(data_name2)
dataset3 = load_dataset(data_name3)

print(dataset1)
print(dataset2)

print(dataset1['train'])
print(dataset2['train'])


# process dataset1
def process_fn(examples):
    examples['instruction']=examples['instruction_zh']
    examples['input']=examples['input_zh']
    examples['output']=examples['output_zh']
    return examples
dataset1 = dataset1.map(process_fn, num_proc=8, remove_columns = ["instruction_zh", "input_zh", 'output_zh'])
dataset1['train'] = dataset1['train'].shard(num_shards=10, index=0)
print(dataset1)

# process dataset2
dataset2 = dataset2.remove_columns([
    'text',
])
print(dataset2)
# dataset2['train'] = dataset2['train'].shard(num_shards=10, index=0)


# process dataset3
def process_ruozhiba(examples):
    examples['input'] = ''
    return examples
dataset3.map(process_ruozhiba, num_proc=8)
print(dataset3)


dataset = concatenate_datasets([dataset1['train'], dataset2['train'], dataset3['train']])
dataset = DatasetDict({'train': dataset})
dataset = dataset.shuffle(seed=42)
print(dataset)
# print(dataset['train'][:20])
for i in range(20):
    print('-'*100)
    print('instruction: ', dataset['train']['instruction'][i])
    print('input: ', dataset['train']['input'][i])
    print('output: ', dataset['train']['output'][i])
dataset.save_to_disk('./output/merge_alpaca_dataset')
