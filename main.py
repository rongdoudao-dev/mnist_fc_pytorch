"""
main.py - 训练入口（全面优化升级版）

优化点（比基础版多了什么）：
  1. 验证集划分（训练集80% + 验证集20%）
  2. 学习率衰减（StepLR，每30epoch×0.5）
  3. 梯度裁剪（max_norm=1.0，防止梯度爆炸）
  4. 早停（Early Stopping，patience=20，验证集loss不下降就停）
  5. 恢复最佳模型（训练结束后自动加载验证集最优的权重）
  6. 数据增强（随机旋转±15°、平移±10%、缩放0.9~1.1，比手动复制粘贴高级）
  7. 梯度监控（每层L2范数、RMS、全局范数裁剪前后）
  8. 丰富可视化（训练曲线、综合仪表盘、混淆矩阵×3、召回率直方图×3、梯度监控、学习率曲线）

用法:
    python main.py
"""
import os
import copy
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from model import FullyConnectedNetwork
from loss import CustomCrossEntropyLoss
from data import HandwrittenDigitsDataset
from metrics import classification_report, compute_accuracy, compute_precision, compute_recall, compute_f1
from utils import (
    set_seed, Timer, AverageMeter,
    plot_training_curves, plot_predictions,
    compute_confusion_matrix, plot_confusion_and_recall,
    plot_gradient_monitoring, plot_learning_rate, plot_full_dashboard,
    plot_management_dashboard,
)


# ============================================================
# 配置超参数
# ============================================================
class Config:
    # 数据
    DATA_ROOT = r"C:\ai727\MNIST_write_num_recog"
    TRAIN_DIR = os.path.join(DATA_ROOT, "training_img")
    TEST_DIR = os.path.join(DATA_ROOT, "test_img")
    IMAGE_SIZE = 32
    VAL_RATIO = 0.2  # 验证集占训练集的比例

    # 模型
    INPUT_SIZE = 32 * 32
    HIDDEN_SIZES = [256, 128, 64]
    NUM_CLASSES = 10

    # 训练
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    LR_STEP_SIZE = 30    # 每多少个epoch衰减一次学习率
    LR_GAMMA = 0.5       # 学习率衰减系数
    GRAD_CLIP_NORM = 1.0 # 梯度裁剪阈值
    EARLY_STOP_PATIENCE = 20  # 早停耐心值

    # 其他
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_DIR = "runs"
    PRINT_EVERY = 10


def compute_gradient_norms(model):
    """计算各层梯度的L2范数和RMS，以及全局梯度范数"""
    layer_norms = []
    layer_rms = []
    total_norm_sq = 0.0

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.detach()
            l2_norm = grad.norm(2).item()
            rms = torch.sqrt(torch.mean(grad ** 2)).item()
            layer_norms.append(l2_norm)
            layer_rms.append(rms)
            total_norm_sq += l2_norm ** 2

    global_norm = np.sqrt(total_norm_sq)
    return layer_norms, layer_rms, global_norm


def train_one_epoch(model, train_loader, criterion, optimizer, device, grad_history, batch_loss_history):
    """训练一个epoch，同时记录梯度、batch损失、所有预测和标签"""
    model.train()
    loss_meter = AverageMeter("Loss")
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    epoch_layer_l2 = []
    epoch_layer_rms = []
    epoch_global_before = []
    epoch_global_after = []

    for batch_images, batch_labels in train_loader:
        batch_images = batch_images.float().to(device)
        batch_labels = batch_labels.long().to(device)

        # 标准5步SOP
        optimizer.zero_grad()
        logits = model(batch_images)
        loss = criterion(logits, batch_labels)
        loss.backward()

        # 梯度裁剪前记录
        layer_l2, layer_rms, global_before = compute_gradient_norms(model)
        epoch_layer_l2.append(layer_l2)
        epoch_layer_rms.append(layer_rms)
        epoch_global_before.append(global_before)

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 梯度裁剪后记录
        _, _, global_after = compute_gradient_norms(model)
        epoch_global_after.append(global_after)

        optimizer.step()

        # 统计
        loss_meter.update(loss.item(), batch_images.size(0))
        batch_loss_history.append(loss.item())
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == batch_labels).sum().item()
        total += batch_labels.size(0)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())

    # 记录本epoch的平均梯度
    if epoch_layer_l2:
        for i in range(len(epoch_layer_l2[0])):
            key_l2 = f"layer_{i}_l2"
            key_rms = f"layer_{i}_rms"
            if key_l2 not in grad_history:
                grad_history[key_l2] = []
                grad_history[key_rms] = []
            grad_history[key_l2].append(np.mean([batch[i] for batch in epoch_layer_l2]))
            grad_history[key_rms].append(np.mean([batch[i] for batch in epoch_layer_rms]))
    grad_history["global_norm_before"].append(np.mean(epoch_global_before))
    grad_history["global_norm_after"].append(np.mean(epoch_global_after))

    return loss_meter.avg, correct / total, np.array(all_predictions), np.array(all_labels)


def evaluate(model, data_loader, criterion, device):
    """评估模型，返回loss、准确率、所有预测、所有标签"""
    model.eval()
    loss_meter = AverageMeter("Loss")
    all_predictions = []
    all_labels = []

    with torch.no_grad():
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
    cfg = Config()
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    set_seed(cfg.SEED)

    print("=" * 70)
    print("手写数字识别 - PyTorch全连接神经网络（全面优化升级版）")
    print("=" * 70)
    print(f"设备: {cfg.DEVICE} | PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"网络: {cfg.INPUT_SIZE} → {' → '.join(map(str, cfg.HIDDEN_SIZES))} → {cfg.NUM_CLASSES}")
    print(f"优化: 数据增强+学习率衰减+梯度裁剪+早停+恢复最佳模型")
    print("=" * 70)

    # ===== 1. 加载数据集（训练集带数据增强，测试集不带）=====
    print("\n[1/6] 加载数据集...")
    full_train_dataset = HandwrittenDigitsDataset(root_dir=cfg.TRAIN_DIR, image_size=cfg.IMAGE_SIZE, augment=True)
    test_dataset = HandwrittenDigitsDataset(root_dir=cfg.TEST_DIR, image_size=cfg.IMAGE_SIZE, augment=False)

    # 划分训练集和验证集（80%训练，20%验证）
    val_size = int(len(full_train_dataset) * cfg.VAL_RATIO)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size],
                                               generator=torch.Generator().manual_seed(cfg.SEED))

    print(f"  训练集: {len(train_dataset)} 张（带数据增强）")
    print(f"  验证集: {len(val_dataset)} 张")
    print(f"  测试集: {len(test_dataset)} 张")

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

    # ===== 2. 创建模型、损失函数、优化器、学习率调度器 =====
    print("\n[2/6] 创建模型、损失函数、优化器...")
    model = FullyConnectedNetwork(
        input_size=cfg.INPUT_SIZE,
        hidden_sizes=cfg.HIDDEN_SIZES,
        num_classes=cfg.NUM_CLASSES
    ).to(cfg.DEVICE)
    print(f"  模型参数量: {model.count_parameters():,}")

    criterion = CustomCrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.LR_STEP_SIZE, gamma=cfg.LR_GAMMA)

    # ===== 3. 训练（含早停、学习率衰减、梯度监控）=====
    print("\n[3/6] 开始训练...")
    history = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "test_loss": [], "test_acc": [],
        "train_precision": [], "train_recall": [], "train_f1": [],
        "val_precision": [], "val_recall": [], "val_f1": [],
    }
    grad_history = {"layer_names": ["隐藏层1-W", "隐藏层1-b", "隐藏层2-W", "隐藏层2-b", "隐藏层3-W", "隐藏层3-b", "输出层-W", "输出层-b"],
                    "global_norm_before": [], "global_norm_after": []}
    batch_loss_history = []
    lr_history = []

    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    timer = Timer()
    timer.start()

    for epoch in range(1, cfg.EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        lr_history.append(current_lr)

        train_loss, train_acc, train_preds_epoch, train_labels_epoch = train_one_epoch(
            model, train_loader, criterion, optimizer, cfg.DEVICE, grad_history, batch_loss_history
        )
        val_loss, val_acc, val_preds_epoch, val_labels_epoch = evaluate(model, val_loader, criterion, cfg.DEVICE)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, cfg.DEVICE)

        scheduler.step()  # 学习率衰减

        # 计算训练集和验证集的P/R/F1（管理层必看指标）
        train_p = compute_precision(train_preds_epoch, train_labels_epoch, cfg.NUM_CLASSES)
        train_r = compute_recall(train_preds_epoch, train_labels_epoch, cfg.NUM_CLASSES)
        train_f1 = compute_f1(train_preds_epoch, train_labels_epoch, cfg.NUM_CLASSES)
        val_p = compute_precision(val_preds_epoch, val_labels_epoch, cfg.NUM_CLASSES)
        val_r = compute_recall(val_preds_epoch, val_labels_epoch, cfg.NUM_CLASSES)
        val_f1 = compute_f1(val_preds_epoch, val_labels_epoch, cfg.NUM_CLASSES)

        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["train_precision"].append(train_p)
        history["train_recall"].append(train_r)
        history["train_f1"].append(train_f1)
        history["val_precision"].append(val_p)
        history["val_recall"].append(val_r)
        history["val_f1"].append(val_f1)

        # 早停 + 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % cfg.PRINT_EVERY == 0 or epoch == cfg.EPOCHS:
            print(f"  Epoch [{epoch:3d}/{cfg.EPOCHS}] lr={current_lr:.5f} "
                  f"| 训练 Loss:{train_loss:.4f} Acc:{train_acc:.2%} P:{train_p:.2%} R:{train_r:.2%} F1:{train_f1:.2%} "
                  f"| 验证 Loss:{val_loss:.4f} Acc:{val_acc:.2%} P:{val_p:.2%} R:{val_r:.2%} F1:{val_f1:.2%} "
                  f"| 测试 Acc:{test_acc:.2%}")

        if patience_counter >= cfg.EARLY_STOP_PATIENCE:
            print(f"\n  早停触发！验证集Loss连续{cfg.EARLY_STOP_PATIENCE}个epoch未下降，停止训练。")
            print(f"  最佳模型在第 {best_epoch} 个epoch，验证Loss={best_val_loss:.4f}")
            break

    timer.stop()
    print(f"\n  训练耗时: {timer.elapsed:.2f} 秒")

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  已恢复第 {best_epoch} 个epoch的最佳模型权重")

    # ===== 4. 最终评估（训练/验证/测试三个集合）=====
    print("\n[4/6] 最终评估（恢复最佳模型后）...")
    train_loss, train_acc, train_preds, train_labels = evaluate(model, train_loader, criterion, cfg.DEVICE)
    val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, cfg.DEVICE)
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, cfg.DEVICE)

    print(f"  训练集: Loss={train_loss:.4f}, Acc={train_acc:.2%}")
    print(f"  验证集: Loss={val_loss:.4f}, Acc={val_acc:.2%}")
    print(f"  测试集: Loss={test_loss:.4f}, Acc={test_acc:.2%}")
    print("\n" + classification_report(test_preds, test_labels, cfg.NUM_CLASSES))

    # ===== 5. 保存模型 =====
    print("\n[5/6] 保存模型...")
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
        "best_epoch": best_epoch,
        "epochs_trained": epoch,
    }, model_path)
    print(f"  模型已保存: {model_path}")

    # ===== 6. 生成所有可视化 =====
    print("\n[6/6] 生成可视化...")

    # 6.1 训练曲线
    plot_training_curves(history, save_path=os.path.join(cfg.SAVE_DIR, "training_curves.png"))

    # 6.2 综合训练仪表盘（6子图）
    plot_full_dashboard(history, grad_history, lr_history, batch_loss_history,
                        save_path=os.path.join(cfg.SAVE_DIR, "training_dashboard.png"))

    # 6.3 混淆矩阵 + 每类召回率直方图（训练/验证/测试）
    cm_train = compute_confusion_matrix(train_preds, train_labels, cfg.NUM_CLASSES)
    cm_val = compute_confusion_matrix(val_preds, val_labels, cfg.NUM_CLASSES)
    cm_test = compute_confusion_matrix(test_preds, test_labels, cfg.NUM_CLASSES)

    plot_confusion_and_recall(cm_train, "训练集", save_path=os.path.join(cfg.SAVE_DIR, "confusion_train.png"))
    plot_confusion_and_recall(cm_val, "验证集", save_path=os.path.join(cfg.SAVE_DIR, "confusion_val.png"))
    plot_confusion_and_recall(cm_test, "测试集", save_path=os.path.join(cfg.SAVE_DIR, "confusion_test.png"))

    # 6.3.1 管理层汇报6图仪表盘（行业标准：泛化/P变化/R变化/Loss/F1/混淆矩阵+P/R）
    plot_management_dashboard(history, cm_test,
                              save_path=os.path.join(cfg.SAVE_DIR, "management_dashboard.png"))

    # 6.4 梯度监控图
    plot_gradient_monitoring(grad_history, save_path=os.path.join(cfg.SAVE_DIR, "gradient_monitoring.png"))

    # 6.5 学习率曲线
    plot_learning_rate(lr_history, save_path=os.path.join(cfg.SAVE_DIR, "learning_rate.png"))

    # 6.6 预测结果图（0-9各一张）
    sample_images, sample_labels, sample_preds, sample_probs = [], [], [], []
    found_digits = set()
    model.eval()
    with torch.no_grad():
        for i in range(len(test_dataset)):
            img, label = test_dataset[i]
            if label in found_digits:
                continue
            found_digits.add(label)
            logits = model(img.unsqueeze(0).to(cfg.DEVICE))
            logits_shifted = logits - logits.max(dim=1, keepdim=True).values
            exp_logits = torch.exp(logits_shifted)
            probs = exp_logits / exp_logits.sum(dim=1, keepdim=True)
            pred = torch.argmax(logits, dim=1).item()
            prob = probs[0, pred].item()
            sample_images.append(img[0].cpu().numpy())
            sample_labels.append(label)
            sample_preds.append(pred)
            sample_probs.append(prob)
            if len(found_digits) == 10:
                break
    plot_predictions(sample_images, sample_labels, sample_preds, sample_probs,
                     save_path=os.path.join(cfg.SAVE_DIR, "predictions.png"))

    print("\n" + "=" * 70)
    print("训练完成！（全面优化升级版）")
    print(f"  最佳epoch: {best_epoch} | 测试准确率: {test_acc:.2%}")
    print(f"  生成的图像:")
    for f in ["training_curves.png", "training_dashboard.png", "management_dashboard.png",
              "confusion_train.png", "confusion_val.png", "confusion_test.png",
              "gradient_monitoring.png", "learning_rate.png", "predictions.png"]:
        print(f"    - runs/{f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
