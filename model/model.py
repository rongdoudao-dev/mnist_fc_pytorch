"""
model.py - 神经网络模型定义
继承 nn.Module，实现 __init__ 和 forward
网络结构: 1024→256→128→64→10 (Sigmoid×3)
"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """模型基类：所有自定义模型继承此类"""

    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播（子类必须实现）"""
        pass

    def count_parameters(self) -> int:
        """统计可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FullyConnectedNetwork(BaseModel):
    """
    全连接神经网络：手写数字识别
    输入: (batch, 1, 32, 32) 灰度图
    输出: (batch, 10) 原始logits（Softmax在损失函数里做）

    结构:
        Linear(1024→256) → Sigmoid
        Linear(256→128)  → Sigmoid
        Linear(128→64)   → Sigmoid
        Linear(64→10)    → 输出（无激活）
    """

    def __init__(self, input_size: int = 32 * 32,
                 hidden_sizes: list = None,
                 num_classes: int = 10):
        """
        参数:
            input_size: 输入维度，默认32×32=1024
            hidden_sizes: 隐藏层神经元数，默认[256, 128, 64]
            num_classes: 输出类别数，默认10（数字0-9）
        """
        super(FullyConnectedNetwork, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        # 构建全连接层（nn.Linear内部就是矩阵相乘 y = x @ W.T + b）
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))  # 线性变换
            layers.append(nn.Sigmoid())                         # Sigmoid激活
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))       # 输出层（无激活）

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（调用 model(x) 时自动执行）
        参数:
            x: 输入图片，shape=(batch_size, 1, 32, 32)
        返回:
            logits: 原始输出，shape=(batch_size, 10)
        """
        # 展平：(batch, 1, 32, 32) → (batch, 1024)
        x = x.view(x.size(0), -1)
        # 全连接层前向传播
        return self.network(x)
