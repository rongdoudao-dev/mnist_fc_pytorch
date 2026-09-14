"""
metric.py - 评估指标计算
包含 Accuracy、Precision、Recall、F1、混淆矩阵、分类报告
"""
import numpy as np
import torch


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    计算准确率
    参数:
        predictions: 预测标签，shape=(N,)
        labels: 真实标签，shape=(N,)
    返回:
        accuracy: 准确率（0-1）
    """
    return np.mean(predictions == labels)


def compute_confusion_matrix(predictions: np.ndarray,
                             labels: np.ndarray,
                             num_classes: int = 10) -> np.ndarray:
    """
    计算混淆矩阵
    参数:
        predictions: 预测标签，shape=(N,)
        labels: 真实标签，shape=(N,)
        num_classes: 类别数
    返回:
        confusion_matrix: shape=(num_classes, num_classes)
            行=真实标签，列=预测标签
    """
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(labels, predictions):
        matrix[true_label][pred_label] += 1
    return matrix


def compute_precision(predictions: np.ndarray,
                      labels: np.ndarray,
                      num_classes: int = 10,
                      average: str = "macro") -> float:
    """
    计算精确率
    参数:
        predictions: 预测标签
        labels: 真实标签
        num_classes: 类别数
        average: "macro"（宏平均）或 "micro"（微平均）
    返回:
        precision: 精确率
    """
    matrix = compute_confusion_matrix(predictions, labels, num_classes)

    if average == "micro":
        tp = np.trace(matrix)
        total = matrix.sum()
        return tp / total if total > 0 else 0.0

    # macro: 每个类别的精确率求平均
    precisions = []
    for i in range(num_classes):
        tp = matrix[i][i]
        fp = matrix[:, i].sum() - tp
        if tp + fp > 0:
            precisions.append(tp / (tp + fp))
    return np.mean(precisions) if precisions else 0.0


def compute_recall(predictions: np.ndarray,
                   labels: np.ndarray,
                   num_classes: int = 10,
                   average: str = "macro") -> float:
    """
    计算召回率
    参数:
        predictions: 预测标签
        labels: 真实标签
        num_classes: 类别数
        average: "macro" 或 "micro"
    返回:
        recall: 召回率
    """
    matrix = compute_confusion_matrix(predictions, labels, num_classes)

    if average == "micro":
        tp = np.trace(matrix)
        total = matrix.sum()
        return tp / total if total > 0 else 0.0

    recalls = []
    for i in range(num_classes):
        tp = matrix[i][i]
        fn = matrix[i, :].sum() - tp
        if tp + fn > 0:
            recalls.append(tp / (tp + fn))
    return np.mean(recalls) if recalls else 0.0


def compute_f1(predictions: np.ndarray,
               labels: np.ndarray,
               num_classes: int = 10,
               average: str = "macro") -> float:
    """
    计算 F1 分数
    F1 = 2 * precision * recall / (precision + recall)
    """
    precision = compute_precision(predictions, labels, num_classes, average)
    recall = compute_recall(predictions, labels, num_classes, average)
    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0


def classification_report(predictions: np.ndarray,
                          labels: np.ndarray,
                          num_classes: int = 10) -> str:
    """
    生成分类报告（类似 sklearn 的 classification_report）
    返回格式化的字符串
    """
    matrix = compute_confusion_matrix(predictions, labels, num_classes)
    accuracy = compute_accuracy(predictions, labels)

    lines = []
    lines.append("=" * 65)
    lines.append(f"{'类别':<6}{'精确率':>10}{'召回率':>10}{'F1':>10}{'支持数':>10}")
    lines.append("-" * 65)

    for i in range(num_classes):
        tp = matrix[i][i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp
        support = matrix[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        lines.append(f"{i:<6}{precision:>10.4f}{recall:>10.4f}{f1:>10.4f}{support:>10d}")

    lines.append("-" * 65)
    macro_precision = compute_precision(predictions, labels, num_classes, "macro")
    macro_recall = compute_recall(predictions, labels, num_classes, "macro")
    macro_f1 = compute_f1(predictions, labels, num_classes, "macro")
    total_support = matrix.sum()

    lines.append(f"{'宏平均':<6}{macro_precision:>10.4f}{macro_recall:>10.4f}{macro_f1:>10.4f}{total_support:>10d}")
    lines.append(f"{'准确率':<6}{accuracy:>10.4f}")
    lines.append("=" * 65)

    return "\n".join(lines)
