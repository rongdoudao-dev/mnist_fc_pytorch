"""
utils 模块：工具函数（升级版）
包含：随机种子、计时器、平均计量器、训练曲线、预测结果、
      混淆矩阵、每类召回率直方图、梯度监控、学习率曲线、综合仪表盘
"""
from .utils import (
    set_seed, Timer, AverageMeter,
    plot_training_curves, plot_predictions,
    compute_confusion_matrix, plot_confusion_matrix,
    plot_per_class_recall, plot_confusion_and_recall,
    plot_gradient_monitoring, plot_learning_rate, plot_full_dashboard,
    plot_management_dashboard, compute_pr_curve, plot_pr_curve,
)

__all__ = [
    "set_seed", "Timer", "AverageMeter",
    "plot_training_curves", "plot_predictions",
    "compute_confusion_matrix", "plot_confusion_matrix",
    "plot_per_class_recall", "plot_confusion_and_recall",
    "plot_gradient_monitoring", "plot_learning_rate", "plot_full_dashboard",
    "plot_management_dashboard", "compute_pr_curve", "plot_pr_curve",
]
