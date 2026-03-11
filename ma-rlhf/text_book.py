from datasets import Dataset
import os

# 定义文本文件所在的文件夹路径
folder_path = "./../dataset/med_qa_textbook"
output_path = './../dataset/second_pretrained_datasets'
texts = []
chunk_size = 512
chunk_mode = True

def split_string(text, chunk_size):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

            if chunk_mode == True:
                text_chunk = split_string(text, chunk_size)
                texts.extend(text_chunk)
                print(file_name, len(text), len(text_chunk))
            else:
                # list mode
                print(file_name, len(text))
                texts.append(text)


print('total:', len(texts))
dataset = Dataset.from_dict({"text": texts})
print('dataset example:', dataset['text'][1][:100])

dataset.save_to_disk(output_path)
