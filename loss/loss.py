"""
loss.py - 自定义损失函数
不直接调用 nn.CrossEntropyLoss，手写 Softmax + 交叉熵
继承 nn.Module，实现 forward 方法
"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseLoss(nn.Module, ABC):
    """损失函数基类：所有自定义损失函数继承此类"""

    def __init__(self):
        super(BaseLoss, self).__init__()

    @abstractmethod
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算损失（子类必须实现）"""
        pass


class CustomCrossEntropyLoss(BaseLoss):
    """
    自定义交叉熵损失函数
    内部包含: 数值稳定的 Softmax 归一化 + 交叉熵计算

    数学公式:
        Softmax: p_i = exp(x_i) / Σ exp(x_j)
        交叉熵: loss = -mean(log(p[正确类别]))

    注意:
        - 减去每行最大值防止 exp 溢出
        - 加 epsilon=1e-8 防止 log(0)
        - 返回 batch 平均损失
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        参数:
            epsilon: 防止 log(0) 的小常数
        """
        super(CustomCrossEntropyLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算交叉熵损失
        参数:
            logits: 模型原始输出，shape=(batch_size, num_classes)
            labels: 真实标签，shape=(batch_size,)，值为类别索引0-9
        返回:
            loss: 标量损失值（batch平均）
        """
        # ===== 步骤1: 数值稳定的 Softmax =====
        # 减去每行最大值，防止 exp 溢出
        logits_shifted = logits - logits.max(dim=1, keepdim=True).values
        # exp 计算
        exp_logits = torch.exp(logits_shifted)
        # 每行求和（归一化分母）
        sum_exp = exp_logits.sum(dim=1, keepdim=True)
        # Softmax 概率
        probabilities = exp_logits / sum_exp  # shape=(batch, 10)，每行和为1

        # ===== 步骤2: 交叉熵计算 =====
        # 取每个样本正确类别的概率
        batch_size = logits.size(0)
        correct_probabilities = probabilities[torch.arange(batch_size), labels]
        # 交叉熵: -log(p)
        negative_log_likelihood = -torch.log(correct_probabilities + self.epsilon)
        # batch 平均
        loss = negative_log_likelihood.mean()

        return loss

    @staticmethod
    def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
        """
        静态方法：计算准确率
        参数:
            logits: 模型输出，shape=(batch, 10)
            labels: 真实标签，shape=(batch,)
        返回:
            accuracy: 准确率（0-1）
        """
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == labels).sum().item()
            total = labels.size(0)
            return correct / total
