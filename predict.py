"""
predict.py - 推理脚本
加载训练好的模型，对单张图片或测试集进行预测和评估

用法:
    python predict.py                    # 评估整个测试集
    python predict.py --image path.png   # 预测单张图片
"""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms

from model import FullyConnectedNetwork
from data import HandwrittenDigitsDataset
from metrics import classification_report, compute_accuracy
from utils import set_seed


def load_model(model_path: str, device: str = "cpu"):
    """
    加载训练好的模型
    参数:
        model_path: 模型权重文件路径
        device: 设备
    返回:
        model: 加载好权重的模型
        config: 模型配置字典
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = FullyConnectedNetwork(
        input_size=config["input_size"],
        hidden_sizes=config["hidden_sizes"],
        num_classes=config["num_classes"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"模型已加载: {model_path}")
    print(f"  训练轮数: {checkpoint.get('epochs', '未知')}")
    print(f"  测试准确率: {checkpoint.get('test_accuracy', '未知'):.2%}")
    return model, config


def predict_single_image(model, image_path: str, config: dict, device: str = "cpu"):
    """
    预测单张图片
    参数:
        model: 模型
        image_path: 图片路径
        config: 模型配置
        device: 设备
    返回:
        result: 预测结果字典
    """
    # 图片预处理（和训练时一致）
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("L")
    image_tensor = transform(image).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        # 手动 softmax 算概率
        logits_shifted = logits - logits.max(dim=1, keepdim=True).values
        exp_logits = torch.exp(logits_shifted)
        probabilities = exp_logits / exp_logits.sum(dim=1, keepdim=True)

        predicted_digit = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0, predicted_digit].item()
        all_probs = probabilities[0].cpu().numpy()

    result = {
        "image_path": image_path,
        "predicted_digit": predicted_digit,
        "confidence": confidence,
        "probabilities": all_probs,
    }

    print(f"\n预测结果:")
    print(f"  图片: {image_path}")
    print(f"  预测数字: {predicted_digit}")
    print(f"  置信度: {confidence:.2%}")
    print(f"  各类别概率:")
    for i, prob in enumerate(all_probs):
        bar = "█" * int(prob * 50)
        print(f"    {i}: {prob:.4f} {bar}")

    return result


def evaluate_test_set(model, test_dir: str, config: dict, device: str = "cpu"):
    """
    评估整个测试集
    参数:
        model: 模型
        test_dir: 测试集文件夹
        config: 模型配置
        device: 设备
    """
    test_dataset = HandwrittenDigitsDataset(root_dir=test_dir, image_size=config["image_size"])
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    all_predictions = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_images = batch_images.float().to(device)
            logits = model(batch_images)
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.numpy())

    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    accuracy = compute_accuracy(predictions, labels)

    print(f"\n测试集评估:")
    print(f"  样本数: {len(labels)}")
    print(f"  准确率: {accuracy:.2%}")
    print(f"\n分类报告:")
    print(classification_report(predictions, labels, config["num_classes"]))


def main():
    parser = argparse.ArgumentParser(description="手写数字识别 - 推理脚本")
    parser.add_argument("--model", type=str, default="runs/model.pth",
                        help="模型权重文件路径")
    parser.add_argument("--image", type=str, default=None,
                        help="单张图片路径（不指定则评估整个测试集）")
    parser.add_argument("--test_dir", type=str,
                        default=r"C:\ai727\MNIST_write_num_recog\test_img",
                        help="测试集文件夹路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("手写数字识别 - 推理")
    print("=" * 60)

    # 加载模型
    model, config = load_model(args.model, device)

    if args.image:
        # 预测单张图片
        predict_single_image(model, args.image, config, device)
    else:
        # 评估整个测试集
        evaluate_test_set(model, args.test_dir, config, device)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
