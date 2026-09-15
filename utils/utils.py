"""
utils.py - 工具函数（升级版）
包含：随机种子、计时器、平均计量器、训练曲线、预测结果、
      混淆矩阵、每类召回率直方图、梯度监控、学习率曲线、综合仪表盘
"""
import os
import random
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 修复中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 自定义蓝色渐变colormap（用于混淆矩阵）
blue_cmap = LinearSegmentedColormap.from_list("custom_blue", ["#f0f8ff", "#1e3a8a"])


def set_seed(seed: int = 42):
    """设置随机种子，保证实验可复现"""
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
        self.start_time = time.time()

    def stop(self) -> float:
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
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self):
        return f"{self.name}: {self.val:.4f} (avg: {self.avg:.4f})"


# ============================================================
# 基础可视化
# ============================================================

def plot_training_curves(history: dict, save_path: str = None):
    """绘制训练曲线（Loss 和 Accuracy，含训练/验证/测试）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="训练集", color="#e74c3c", linewidth=2)
    if "val_loss" in history:
        ax1.plot(history["val_loss"], label="验证集", color="#f39c12", linewidth=2)
    ax1.plot(history["test_loss"], label="测试集", color="#3498db", linewidth=2)
    ax1.set_title("损失曲线（看拟合差距）", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("交叉熵损失")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["train_acc"], label="训练集", color="#e74c3c", linewidth=2)
    if "val_acc" in history:
        ax2.plot(history["val_acc"], label="验证集", color="#f39c12", linewidth=2)
    ax2.plot(history["test_acc"], label="测试集", color="#3498db", linewidth=2)
    ax2.set_title("准确率曲线", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("准确率")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"训练曲线已保存: {save_path}")
    plt.close(fig)


def plot_predictions(images: list, labels: list, predictions: list,
                     probabilities: list, save_path: str = None):
    """绘制预测结果网格图（0-9各一张）"""
    num_images = len(images)
    cols = 5
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten()

    for i, (img, label, pred, prob) in enumerate(zip(images, labels, predictions, probabilities)):
        axes[i].imshow(img, cmap="gray")
        color = "#27ae60" if label == pred else "#e74c3c"
        axes[i].set_title(f"真实:{label} | 预测:{pred}\n置信度:{prob:.1%}",
                          color=color, fontsize=11, fontweight="bold")
        axes[i].axis("off")

    for i in range(num_images, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"预测结果图已保存: {save_path}")
    plt.close(fig)


# ============================================================
# 混淆矩阵 + 每类召回率直方图
# ============================================================

def compute_confusion_matrix(predictions: np.ndarray, labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """计算混淆矩阵"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for pred, label in zip(predictions, labels):
        cm[label][pred] += 1
    return cm


def plot_confusion_matrix(cm: np.ndarray, title: str = "混淆矩阵", save_path: str = None):
    """绘制混淆矩阵（蓝色渐变，带数值标注）"""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=blue_cmap)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=[str(i) for i in range(cm.shape[1])],
           yticklabels=[str(i) for i in range(cm.shape[0])])
    ax.set_xlabel("预测类别", fontsize=12)
    ax.set_ylabel("真实类别", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # 在每个格子里写数值
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center", color=color, fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"混淆矩阵已保存: {save_path}")
    plt.close(fig)


def plot_per_class_recall(cm: np.ndarray, title: str = "每个数字的识别情况", save_path: str = None):
    """绘制每类召回率直方图（带样本数标注）"""
    num_classes = cm.shape[0]
    recalls = []
    counts = []
    for i in range(num_classes):
        total = cm[i].sum()
        counts.append(total)
        recalls.append(cm[i][i] / total if total > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(num_classes), recalls, color="#2c7873", edgecolor="#1a5c57", linewidth=1.2)
    ax.set_xlabel("真实类别", fontsize=12)
    ax.set_ylabel("该类召回率", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(range(num_classes))
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3, axis="y")

    # 在柱子上标注样本数和召回率
    for bar, count, recall in zip(bars, counts, recalls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={count}\n{recall:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"每类召回率直方图已保存: {save_path}")
    plt.close(fig)


def plot_confusion_and_recall(cm: np.ndarray, dataset_name: str, save_path: str = None):
    """在同一张图里绘制混淆矩阵 + 每类召回率直方图（左右排列）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 左：混淆矩阵
    im = ax1.imshow(cm, interpolation="nearest", cmap=blue_cmap)
    ax1.set_xticks(range(10))
    ax1.set_yticks(range(10))
    ax1.set_xlabel("预测类别", fontsize=11)
    ax1.set_ylabel("真实类别", fontsize=11)
    total = cm.sum()
    correct = np.trace(cm)
    ax1.set_title(f"{dataset_name}混淆矩阵\n{correct}/{total} = {correct/total:.2%}", fontsize=13, fontweight="bold")
    thresh = cm.max() / 2.0
    for i in range(10):
        for j in range(10):
            color = "white" if cm[i, j] > thresh else "black"
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=9)

    # 右：每类召回率直方图
    recalls = [cm[i][i] / cm[i].sum() if cm[i].sum() > 0 else 0 for i in range(10)]
    counts = [cm[i].sum() for i in range(10)]
    bars = ax2.bar(range(10), recalls, color="#2c7873", edgecolor="#1a5c57")
    ax2.set_xlabel("真实类别", fontsize=11)
    ax2.set_ylabel("该类召回率", fontsize=11)
    ax2.set_title(f"{dataset_name}每个数字的识别情况", fontsize=13, fontweight="bold")
    ax2.set_xticks(range(10))
    ax2.set_ylim(0, 1.2)
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, count, recall in zip(bars, counts, recalls):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"n={count}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"混淆矩阵+召回率图已保存: {save_path}")
    plt.close(fig)


# ============================================================
# 梯度监控 + 学习率曲线
# ============================================================

def plot_gradient_monitoring(grad_history: dict, save_path: str = None):
    """
    绘制梯度监控图（3子图）：
    1. 各层梯度L2范数（裁剪前）
    2. 各层梯度RMS（对数坐标，比较不同宽度）
    3. 全模型梯度范数（裁剪前后对比）
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["#3498db", "#f39c12", "#27ae60"]
    layer_names = grad_history.get("layer_names", ["隐藏层1", "隐藏层2", "输出层"])

    # 子图1：各层梯度L2范数
    for i, name in enumerate(layer_names):
        key = f"layer_{i}_l2"
        if key in grad_history:
            axes[0].plot(grad_history[key], label=name, color=colors[i % len(colors)], linewidth=1.5)
    axes[0].set_title("各层梯度L2范数（裁剪前）", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("L2范数")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 子图2：各层梯度RMS（对数坐标）
    for i, name in enumerate(layer_names):
        key = f"layer_{i}_rms"
        if key in grad_history:
            axes[1].plot(grad_history[key], label=name, color=colors[i % len(colors)], linewidth=1.5)
    axes[1].set_title("各层梯度RMS（对数坐标，比较不同宽度）", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMS（对数）")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 子图3：全模型梯度范数（裁剪前后）
    if "global_norm_before" in grad_history:
        axes[2].plot(grad_history["global_norm_before"], label="裁剪前", color="#e74c3c", linewidth=1.5)
    if "global_norm_after" in grad_history:
        axes[2].plot(grad_history["global_norm_after"], label="裁剪后", color="#3498db", linewidth=1.5)
    axes[2].set_title("全模型梯度范数（看裁剪影响）", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("全局范数")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"梯度监控图已保存: {save_path}")
    plt.close(fig)


def plot_learning_rate(lr_history: list, save_path: str = None):
    """绘制学习率衰减曲线"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lr_history, color="#9b59b6", linewidth=2)
    ax.set_title("学习率变化曲线", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("学习率")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"学习率曲线已保存: {save_path}")
    plt.close(fig)


# ============================================================
# 综合训练仪表盘（6子图，类似同学的专业版）
# ============================================================

def plot_full_dashboard(history: dict, grad_history: dict, lr_history: list,
                        batch_loss_history: list, save_path: str = None):
    """
    综合训练仪表盘（2行×3列 = 6子图）：
    1. 训练/验证损失
    2. 训练/验证准确率
    3. 每次参数更新前的batch损失
    4. 各层梯度L2范数
    5. 各层梯度RMS（对数）
    6. 全模型梯度范数（裁剪前后）
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = ["#3498db", "#f39c12", "#27ae60"]
    layer_names = grad_history.get("layer_names", ["隐藏层1", "隐藏层2", "输出层"])

    # 1. 损失曲线
    axes[0, 0].plot(history["train_loss"], label="训练集", color="#e74c3c", linewidth=1.5)
    if "val_loss" in history:
        axes[0, 0].plot(history["val_loss"], label="验证集", color="#f39c12", linewidth=1.5)
    axes[0, 0].set_title("同一评估模式的损失：看拟合差距", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("交叉熵")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 准确率曲线
    axes[0, 1].plot(history["train_acc"], label="训练集", color="#e74c3c", linewidth=1.5)
    if "val_acc" in history:
        axes[0, 1].plot(history["val_acc"], label="验证集", color="#f39c12", linewidth=1.5)
    axes[0, 1].set_title("训练集与验证集准确率", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("正确数/总数")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. batch损失（每次参数更新前）
    axes[0, 2].plot(batch_loss_history, color="#16a085", linewidth=0.5, alpha=0.7)
    axes[0, 2].set_title("每次参数更新前的batch损失", fontsize=11, fontweight="bold")
    axes[0, 2].set_xlabel("batch迭代次数（不同批次允许波动）")
    axes[0, 2].set_ylabel("训练模式交叉熵")
    axes[0, 2].grid(True, alpha=0.3)

    # 4. 各层梯度L2范数
    for i, name in enumerate(layer_names):
        key = f"layer_{i}_l2"
        if key in grad_history:
            axes[1, 0].plot(grad_history[key], label=name, color=colors[i % len(colors)], linewidth=1.5)
    axes[1, 0].set_title("各层梯度L2范数：裁剪前", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("每轮各batch平均；零值画在1e-12")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. 各层梯度RMS（对数）
    for i, name in enumerate(layer_names):
        key = f"layer_{i}_rms"
        if key in grad_history:
            axes[1, 1].plot(grad_history[key], label=name, color=colors[i % len(colors)], linewidth=1.5)
    axes[1, 1].set_title("各层梯度RMS：帮助比较不同宽度", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("每轮各batch平均；零值画在1e-12")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. 全模型梯度范数（裁剪前后）
    if "global_norm_before" in grad_history:
        axes[1, 2].plot(grad_history["global_norm_before"], label="裁剪前", color="#e74c3c", linewidth=1.5)
    if "global_norm_after" in grad_history:
        axes[1, 2].plot(grad_history["global_norm_after"], label="裁剪后", color="#3498db", linewidth=1.5)
    axes[1, 2].set_title("全模型梯度范数：看裁剪影响", fontsize=11, fontweight="bold")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("全局范数")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"综合训练仪表盘已保存: {save_path}")
    plt.close(fig)
