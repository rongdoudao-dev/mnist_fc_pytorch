"""
main.py - 训练入口（工程化作业）
配置超参数 → 加载数据 → 创建模型/损失/优化器 → 训练 → 评估 → 保存 → 可视化

用法:
    python main.py

训练结束后:
    - 保存 runs/training_curves.png（训练曲线）
    - 保存 runs/predictions.png（预测结果）
    - 保存 runs/model.pth（模型权重）
    - 打印测试集分类报告（准确率/精确率/召回率/F1/混淆矩阵）
"""
import os
import sys
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# 导入自定义模块
from model import FullyConnectedNetwork
from loss import CustomCrossEntropyLoss
from data import HandwrittenDigitsDataset
from metrics import classification_report, compute_accuracy
from utils import set_seed, Timer, AverageMeter, plot_training_curves, plot_predictions


# ============================================================
# 配置超参数（顶部配置区块）
# ============================================================
class Config:
    """超参数配置"""
    # 数据
    DATA_ROOT = r"C:\ai727\MNIST_write_num_recog"
    TRAIN_DIR = os.path.join(DATA_ROOT, "training_img")
    TEST_DIR = os.path.join(DATA_ROOT, "test_img")
    IMAGE_SIZE = 32

    # 模型
    INPUT_SIZE = 32 * 32  # 1024
    HIDDEN_SIZES = [256, 128, 64]
    NUM_CLASSES = 10

    # 训练
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    PRINT_EVERY = 10  # 每多少个epoch打印一次

    # 其他
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_DIR = "runs"


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    训练一个 epoch
    返回: (平均loss, 准确率)
    """
    model.train()
    loss_meter = AverageMeter("Loss")
    correct = 0
    total = 0

    # 小for：遍历 batch
    for batch_images, batch_labels in train_loader:
        batch_images = batch_images.float().to(device)
        batch_labels = batch_labels.long().to(device)

        # 标准 SOP 5 步
        optimizer.zero_grad()                    # ① 清零梯度
        logits = model(batch_images)              # ② 前向传播
        loss = criterion(logits, batch_labels)    # ③ 计算损失
        loss.backward()                            # ④ 反向传播（自动算梯度）
        optimizer.step()                           # ⑤ 更新参数

        # 统计
        loss_meter.update(loss.item(), batch_images.size(0))
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == batch_labels).sum().item()
        total += batch_labels.size(0)

    return loss_meter.avg, correct / total


def evaluate(model, data_loader, criterion, device):
    """
    评估模型
    返回: (平均loss, 准确率, 所有预测, 所有标签)
    """
    model.eval()
    loss_meter = AverageMeter("Loss")
    all_predictions = []
    all_labels = []

    with torch.no_grad():  # 评估时不计算梯度
        for batch_images, batch_labels in data_loader:
            batch_images = batch_images.float().to(device)
            batch_labels = batch_labels.long().to(device)

            logits = model(batch_images)
            loss = criterion(logits, batch_labels)

            loss_meter.update(loss.item(), batch_images.size(0))
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    accuracy = compute_accuracy(np.array(all_predictions), np.array(all_labels))
    return loss_meter.avg, accuracy, np.array(all_predictions), np.array(all_labels)


def main():
    """主训练流程"""
    cfg = Config()

    # 创建保存目录
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)

    # 设置随机种子（保证可复现）
    set_seed(cfg.SEED)

    print("=" * 70)
    print("手写数字识别 - PyTorch全连接神经网络（工程化作业）")
    print("=" * 70)
    print(f"设备: {cfg.DEVICE}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"随机种子: {cfg.SEED}")
    print(f"网络结构: {cfg.INPUT_SIZE} → {' → '.join(map(str, cfg.HIDDEN_SIZES))} → {cfg.NUM_CLASSES}")
    print(f"Batch Size: {cfg.BATCH_SIZE} | Epochs: {cfg.EPOCHS} | LR: {cfg.LEARNING_RATE}")
    print("=" * 70)

    # ===== 1. 加载数据集 =====
    print("\n[1/5] 加载数据集...")
    train_dataset = HandwrittenDigitsDataset(root_dir=cfg.TRAIN_DIR, image_size=cfg.IMAGE_SIZE)
    test_dataset = HandwrittenDigitsDataset(root_dir=cfg.TEST_DIR, image_size=cfg.IMAGE_SIZE)
    print(f"  训练集: {len(train_dataset)} 张")
    print(f"  测试集: {len(test_dataset)} 张")

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

    # ===== 2. 创建模型、损失函数、优化器 =====
    print("\n[2/5] 创建模型、损失函数、优化器...")
    model = FullyConnectedNetwork(
        input_size=cfg.INPUT_SIZE,
        hidden_sizes=cfg.HIDDEN_SIZES,
        num_classes=cfg.NUM_CLASSES
    ).to(cfg.DEVICE)
    print(f"  模型参数量: {model.count_parameters():,}")

    criterion = CustomCrossEntropyLoss()  # 自定义损失函数（手写Softmax+交叉熵）
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        momentum=cfg.MOMENTUM
    )
    print(f"  损失函数: {criterion.__class__.__name__}")
    print(f"  优化器: SGD (lr={cfg.LEARNING_RATE}, momentum={cfg.MOMENTUM})")

    # ===== 3. 训练 =====
    print("\n[3/5] 开始训练...")
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    timer = Timer()
    timer.start()

    # 大for：遍历 epoch
    for epoch in range(1, cfg.EPOCHS + 1):
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, cfg.DEVICE, epoch
        )
        # 评估
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, cfg.DEVICE)

        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        # 每 PRINT_EVERY 个 epoch 打印一次（第1个也打印）
        if epoch == 1 or epoch % cfg.PRINT_EVERY == 0 or epoch == cfg.EPOCHS:
            print(f"  Epoch [{epoch:3d}/{cfg.EPOCHS}] "
                  f"| 训练 Loss: {train_loss:.4f}  Acc: {train_acc:.2%} "
                  f"| 测试 Loss: {test_loss:.4f}  Acc: {test_acc:.2%}")

    timer.stop()
    print(f"\n  训练耗时: {timer.elapsed:.2f} 秒")

    # ===== 4. 最终评估 =====
    print("\n[4/5] 最终评估...")
    test_loss, test_acc, test_predictions, test_labels = evaluate(
        model, test_loader, criterion, cfg.DEVICE
    )
    print(f"  最终测试 Loss: {test_loss:.4f}")
    print(f"  最终测试准确率: {test_acc:.2%}")

    # 打印分类报告
    print("\n" + classification_report(test_predictions, test_labels, cfg.NUM_CLASSES))

    # ===== 5. 保存模型和可视化 =====
    print("\n[5/5] 保存模型和可视化...")

    # 保存模型权重
    model_path = os.path.join(cfg.SAVE_DIR, "model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "input_size": cfg.INPUT_SIZE,
            "hidden_sizes": cfg.HIDDEN_SIZES,
            "num_classes": cfg.NUM_CLASSES,
            "image_size": cfg.IMAGE_SIZE,
        },
        "test_accuracy": test_acc,
        "epochs": cfg.EPOCHS,
    }, model_path)
    print(f"  模型已保存: {model_path}")

    # 绘制训练曲线
    curves_path = os.path.join(cfg.SAVE_DIR, "training_curves.png")
    plot_training_curves(history, save_path=curves_path)

    # 绘制预测结果（取前8张测试样本）
    sample_images, sample_labels, sample_preds, sample_probs = [], [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(min(8, len(test_dataset))):
            img, label = test_dataset[i]
            logits = model(img.unsqueeze(0).to(cfg.DEVICE))
            # 手动 softmax 算概率
            logits_shifted = logits - logits.max(dim=1, keepdim=True).values
            exp_logits = torch.exp(logits_shifted)
            probs = exp_logits / exp_logits.sum(dim=1, keepdim=True)
            pred = torch.argmax(logits, dim=1).item()
            prob = probs[0, pred].item()

            sample_images.append(img[0].cpu().numpy())
            sample_labels.append(label)
            sample_preds.append(pred)
            sample_probs.append(prob)

    preds_path = os.path.join(cfg.SAVE_DIR, "predictions.png")
    plot_predictions(sample_images, sample_labels, sample_preds, sample_probs, save_path=preds_path)

    print("\n" + "=" * 70)
    print("训练完成！")
    print(f"  最终测试准确率: {test_acc:.2%}")
    print(f"  模型保存: {model_path}")
    print(f"  训练曲线: {curves_path}")
    print(f"  预测结果: {preds_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
