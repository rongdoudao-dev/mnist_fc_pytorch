"""
loss 模块：损失函数定义
包含 BaseLoss 基类和 CustomCrossEntropyLoss 自定义交叉熵损失
"""
from .loss import BaseLoss, CustomCrossEntropyLoss

__all__ = ["BaseLoss", "CustomCrossEntropyLoss"]
