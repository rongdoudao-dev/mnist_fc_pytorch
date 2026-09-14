"""
metrics 模块：评估指标
包含 Accuracy、Precision、Recall、F1、混淆矩阵等计算函数
"""
from .metric import (
    compute_accuracy,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_confusion_matrix,
    classification_report,
)

__all__ = [
    "compute_accuracy",
    "compute_precision",
    "compute_recall",
    "compute_f1",
    "compute_confusion_matrix",
    "classification_report",
]
