# 手写数字识别 - PyTorch全连接神经网络（工程化作业）

使用 PyTorch 从零实现手写数字识别，包含自定义模型、自定义损失函数、数据集处理、评估指标、训练可视化与模型保存/加载。

## 项目结构

```
mnist_fc_pytorch/
├── main.py                  # 训练入口：配置超参数、训练、评估、保存模型、可视化
├── predict.py               # 推理脚本：加载模型做预测与评估
├── requirements.txt         # 依赖清单
├── README.md                # 项目说明
├── model/
│   ├── __init__.py
│   └── model.py             # BaseModel 基类 + FullyConnectedNetwork 全连接网络
├── loss/
│   ├── __init__.py
│   └── loss.py              # BaseLoss 基类 + CustomCrossEntropyLoss 自定义交叉熵
├── data/
│   ├── __init__.py
│   └── dataset.py           # HandwrittenDigitsDataset 自定义数据集
├── metrics/
│   ├── __init__.py
│   └── metric.py            # Accuracy/Precision/Recall/F1/混淆矩阵/分类报告
├── utils/
│   ├── __init__.py
│   └── utils.py             # set_seed/Timer/AverageMeter/可视化
└── runs/                    # 训练输出（模型权重、曲线图、预测图）
    ├── model.pth            # 训练后保存的模型权重
    ├── training_curves.png  # 训练 Loss + Accuracy 曲线
    └── predictions.png      # 测试样本预测结果
```

## 环境要求

- Python 3.9+
- PyTorch（支持 CUDA 加速）
- torchvision
- NumPy
- Matplotlib
- Pillow

```bash
pip install -r requirements.txt
```

## 数据集

MNIST 手写数字数据集（32×32 灰度图）：
- 训练集：1934 张
- 测试集：946 张
- 文件名格式：`数字_序号.png`（如 `3_045.png`）

数据集路径在 `main.py` 顶部 `Config.DATA_ROOT` 中配置。

## 快速开始

### 1. 训练

```bash
python main.py
```

按 `main.py` 顶部 `Config` 配置区块选择超参数，训练结束后：

- 保存 `runs/training_curves.png`（训练 Loss + Accuracy 曲线）
- 保存 `runs/predictions.png`（8 张测试样本预测结果）
- 保存 `runs/model.pth`（模型权重 + 优化器状态 + 配置）
- 打印测试集分类报告（准确率/精确率/召回率/F1/混淆矩阵）
- 每 10 个 epoch 打印一次训练/测试 Loss 和准确率

### 2. 推理

```bash
# 评估整个测试集
python predict.py

# 预测单张图片
python predict.py --image path/to/image.png
```

加载 `runs/model.pth`，打印预测结果、置信度和各类别概率分布。

## 配置说明（main.py 顶部 Config）

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `BATCH_SIZE` | 64 | 小批量梯度下降的 batch 大小 |
| `EPOCHS` | 100 | 训练轮数 |
| `LEARNING_RATE` | 0.1 | 学习率 |
| `MOMENTUM` | 0.9 | SGD 动量系数 |
| `PRINT_EVERY` | 10 | 每多少轮打印一次日志 |
| `SEED` | 42 | 随机种子，保证可复现 |
| `HIDDEN_SIZES` | [256, 128, 64] | 隐藏层神经元数 |

## 网络结构

```
输入: (batch, 1, 32, 32) 灰度图
    ↓ 展平
Linear(1024 → 256) → Sigmoid
Linear(256 → 128)  → Sigmoid
Linear(128 → 64)   → Sigmoid
Linear(64 → 10)    → 输出（无激活，Softmax 在损失函数里）
    ↓
输出: (batch, 10) 原始 logits
```

总参数量：约 304,202

## 核心设计（工业级标准）

### 1. 模型与损失解耦（对齐 PyTorch 风格）

```python
# 标准 SOP 5 步
optimizer.zero_grad()       # ① 清零梯度
logits = model(batch_x)     # ② 前向传播
loss = criterion(logits, y) # ③ 计算损失
loss.backward()             # ④ 反向传播（自动算梯度）
optimizer.step()            # ⑤ 更新参数
```

### 2. 自定义损失函数（不直接用 nn.CrossEntropyLoss）

`CustomCrossEntropyLoss` 内部手写实现：
- 数值稳定的 Softmax（减去每行最大值防止 exp 溢出）
- 交叉熵计算（加 epsilon 防止 log(0)）
- 返回 batch 平均损失

### 3. 模块化设计

- `model/`：模型定义（基类 + 具体模型）
- `loss/`：损失函数（基类 + 具体损失）
- `data/`：数据集处理
- `metrics/`：评估指标
- `utils/`：工具函数

### 4. 可复现性

- 固定随机种子（Python / NumPy / PyTorch / CUDA）
- 保存模型权重 + 优化器状态 + 配置信息
- 训练曲线和预测结果可视化

## 训练结果

| 指标 | 值 |
|---|---|
| 训练集准确率 | ~100% |
| 测试集准确率 | ~97% |
| 训练耗时 | ~10 秒（GPU） |
| 设备 | NVIDIA RTX 5060（CUDA 12.8） |

## 技术要点

1. **维度匹配**：输入层节点数 = 图像展平后像素数（32×32=1024）
2. **梯度清零**：每次迭代前必须 `optimizer.zero_grad()`，防止梯度累积
3. **设备一致**：模型和数据必须在同一设备上（都 CPU 或都 GPU）
4. **评估模式**：评估时用 `model.eval()` + `torch.no_grad()`
5. **数值稳定**：Softmax 计算时减去最大值防止溢出
6. **fp32 精度**：PyTorch 默认 float32 精度

## 许可证

MIT License
