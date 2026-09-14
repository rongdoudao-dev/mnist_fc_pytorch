"""
utils.py - 工具函数
包含随机种子设置、计时器、平均计量器、训练曲线绘制、预测结果可视化
"""
import os
import random
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt

# 修复中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def set_seed(seed: int = 42):
    """
    设置随机种子，保证实验可复现
    参数:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Timer:
    """计时器：统计代码运行时间"""

    def __init__(self):
        self.start_time = None
        self.elapsed = 0.0

    def start(self):
        """开始计时"""
        self.start_time = time.time()

    def stop(self) -> float:
        """停止计时，返回经过的时间（秒）"""
        if self.start_time is not None:
            self.elapsed = time.time() - self.start_time
            self.start_time = None
        return self.elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class AverageMeter:
    """平均计量器：统计损失、准确率等指标的滑动平均"""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        """重置所有统计值"""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        更新统计值
        参数:
            val: 当前值
            n: 样本数
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self):
        return f"{self.name}: {self.val:.4f} (avg: {self.avg:.4f})"


def plot_training_curves(history: dict, save_path: str = None):
    """
    绘制训练曲线（Loss 和 Accuracy）
    参数:
        history: 训练历史字典，包含 train_loss, train_acc, test_loss, test_acc
        save_path: 图片保存路径，None 则不保存
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss 曲线
    ax1.plot(history["train_loss"], label="Train Loss", color="#e74c3c", linewidth=2)
    ax1.plot(history["test_loss"], label="Test Loss", color="#3498db", linewidth=2)
    ax1.set_title("Loss Curve", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy 曲线
    ax2.plot(history["train_acc"], label="Train Accuracy", color="#e74c3c", linewidth=2)
    ax2.plot(history["test_acc"], label="Test Accuracy", color="#3498db", linewidth=2)
    ax2.set_title("Accuracy Curve", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"训练曲线已保存: {save_path}")
    plt.close(fig)


def plot_predictions(images: list, labels: list, predictions: list,
                     probabilities: list, save_path: str = None):
    """
    绘制预测结果网格图
    参数:
        images: 图片列表（numpy数组）
        labels: 真实标签列表
        predictions: 预测标签列表
        probabilities: 预测置信度列表
        save_path: 图片保存路径
    """
    num_images = len(images)
    cols = 4
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten()

    for i, (img, label, pred, prob) in enumerate(zip(images, labels, predictions, probabilities)):
        axes[i].imshow(img, cmap="gray")
        color = "#27ae60" if label == pred else "#e74c3c"
        axes[i].set_title(f"真实:{label} | 预测:{pred}\n置信度:{prob:.1%}",
                          color=color, fontsize=10, fontweight="bold")
        axes[i].axis("off")

    for i in range(num_images, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"预测结果图已保存: {save_path}")
    plt.close(fig)
