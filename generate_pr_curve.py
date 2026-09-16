"""
generate_pr_curve.py - 快速生成P-R曲线（加载已有模型，不重新训练）
用法: python generate_pr_curve.py
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import FullyConnectedNetwork
from data import HandwrittenDigitsDataset
from utils import plot_pr_curve, set_seed


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # 1. 加载测试集
    test_dir = r"C:\ai727\MNIST_write_num_recog\test_img"
    test_dataset = HandwrittenDigitsDataset(root_dir=test_dir, image_size=32, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    print(f"测试集: {len(test_dataset)} 张")

    # 2. 加载已有模型
    model = FullyConnectedNetwork(input_size=1024, hidden_sizes=[256, 128, 64], num_classes=10).to(device)
    checkpoint = torch.load(os.path.join("runs", "model.pth"), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"已加载模型（测试准确率: {checkpoint['test_accuracy']:.2%}）")

    # 3. 推理收集预测概率
    all_probabilities = []
    all_labels = []
    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_images = batch_images.float().to(device)
            logits = model(batch_images)
            # 数值稳定的Softmax
            logits_shifted = logits - logits.max(dim=1, keepdim=True).values
            exp_logits = torch.exp(logits_shifted)
            probabilities = exp_logits / exp_logits.sum(dim=1, keepdim=True)
            all_probabilities.append(probabilities.cpu().numpy())
            all_labels.append(batch_labels.numpy())

    all_probabilities = np.concatenate(all_probabilities, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(f"收集到 {len(all_labels)} 个样本的预测概率")

    # 4. 画P-R曲线
    plot_pr_curve(all_probabilities, all_labels, num_classes=10,
                  save_path=os.path.join("runs", "pr_curve.png"))

    print("\n完成！P-R曲线已保存到 runs/pr_curve.png")


if __name__ == "__main__":
    main()
