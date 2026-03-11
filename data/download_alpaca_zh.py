# 1. download data

import requests
import os

# URL
url = "https://raw.githubusercontent.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/main/data/alpaca_gpt4_data_zh.json"

# 目标文件路径
output_dir = "./output"
output_file = os.path.join(output_dir, "alpaca_gpt4_data_zh.json")
print(output_dir)

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 发起请求并保存文件
response = requests.get(url)
with open(output_file, "wb") as file:
    file.write(response.content)

print("文件下载完成。")


