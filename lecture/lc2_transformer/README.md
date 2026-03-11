# Trasformer

本节内容从零写 Transformer 训练。

- 实现基础的 Transformer 训练
- 掌握配套组件 Dataset、Tokenizer、IO、config 等实现细节。 以帮助更好理解主流框架的实现思路。

本章课件先在 Notebook 调试，再组织成 `.py` 可以用于工程级别训练。

- ✅ ：必读
- 🌟 ：重点学习代码，最好能够独立手撕

## Notebook

初学者在学习 Transformer 前需要思考两个问题：

- 语言是如何被机器理解和表示的？
- 如何根据语义信息进行翻译

|      | 文件名                        | 介绍                                                         | 必读 |
| ---- | ----------------------------- | ------------------------------------------------------------ | ---- |
|      | `Transformer_Framework.ipynb` | 从序列建模角度实现 Encoder-Decoder 机器翻译，配备训练和推理极简代码 | ✅    |
|      | `Transformer_Attention.ipynb` | 序列语言模型的具体实现：注意力机制。语言模型学习 token 在上下文中的表示，从而刻画完整的语义信息。额外增加 手撕注意力 backward。 | ✅🌟   |
|      | `Position_Encoding.ipynb`     | 代码实现和分析                                                         |  ✅🌟   |
|      | `LayerNorm.ipynb`             | 算法原理、可视化、反向                                       | ✅    |
|      | `Transformer.ipynb`           | 完整的 Transformer 模型                                      | ✅🌟   |
|      | `Load_Dataset.ipynb`          | 从 huggingface 加载一个 机器翻译数据集 `wmt/wmt19` , 并存储成 `./data.json`。数据集会用于实际训练实践中。如无法拉取可以先跳过。 |      |
|      | `Dataset.ipynb`               | 将数据集封装成 `Dataset`, 并配套实现 `DataLoader`、`DataCollate` , 用于 batch 加载。 接口参考 `Transformer` 库，实现 mask, padding, label 等的处理技巧 | ✅    |
|      | `Model_IO.ipynb`              | 训练模型完毕后进行保存，将保存的模型加载用于推理。在实现上增加对 优化器 参数的存储，能够了解 checkpoint 断点恢复的训练思路。另外有助于理解如何通过 config 来加载一个模型，这是 `Transformer` 库常用的加载方式 | ✅    |

## Code

当我们理解 Transformer 后，训练模型是容易的。需要能手撕完整的 model 和 train。另外，增加工程实现问题：

1. tokenizer IO 哪些东西？
2. 如何从 config 加载 model？
3. 机器翻译数据如何构建，何时 padding 和 mask？
4. 如何断点训练？
5. 如何绘制训练曲线？

| 文件名         | 介绍                                                         | 必读 |
| -------------- | ------------------------------------------------------------ | ---- |
| `tokenizer.py` | 加载`data.json`中英语料，分别训练中英分词器，可以保存和加载。可以了解分词器存储的内容和分词的规则，共同决定 encode 和 decode 结果。 | ✅    |
| `dataset.py`   | 将 `data.json` 转化为 训练所用的 数据格式。                  | ✅    |
| `utils.py`     | 处理 mask 和 参数对象                                        | ✅    |
| `config.py`    | 独立管理 config, 可通过 config 初始化模型                    | ✅    |
| `model.py`     | 实现 Transformer 模型，需注意每个模块的输入参数。注意 mask 的实现细节。 | ✅🌟   |
| `train.py`     | 加载 `tokenizer`, `dataset` ，并初始化一个 Transformer 模型进行训练，在训练中用交叉熵损失，手写训练流程。TODO：训练曲线绘制。 | ✅🌟   |
| `inference.py` | 加载已训练好的模型，用于推理预测。即中英翻译测试。           | ✅🌟   |

## 训练实践

首先数据可以从 huggingface 拉取，并保存成 `./data.json`, 本目录下已有现成数据，受网络环境影响无法拉取，则可以跳过。

```bash
Load_Dataset.ipynb # optional
```

按照以下顺序, 可以完成 分词器 和 Transformer的训练和推理实践

```bash
python tokenizer.py
python train.py --learning_rate 1e-4 --epochs 1
python inference.py
```

## 扩展知识点

- attention score 除于 sqrt(d)
- softmax + cross-entropy forward & backward
- adam
- auto-grad
- mlp backward
- multi-head attention backward