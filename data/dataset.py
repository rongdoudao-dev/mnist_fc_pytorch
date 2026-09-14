"""
dataset.py - 自定义数据集处理
继承 torch.utils.data.Dataset，实现 __len__ 和 __getitem__
从文件夹加载手写数字图片，文件名格式 "数字_序号.png"
"""
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class HandwrittenDigitsDataset(Dataset):
    """
    手写数字数据集：从文件夹加载图片
    文件名格式: "数字_序号.png"（如 "3_045.png"）

    用法:
        dataset = HandwrittenDigitsDataset(root_dir="training_img")
        image, label = dataset[0]  # image.shape=(1,32,32), label=数字
        print(len(dataset))         # 样本总数
    """

    def __init__(self, root_dir: str, image_size: int = 32, transform=None):
        """
        参数:
            root_dir: 图片文件夹路径
            image_size: 图片缩放尺寸，默认32×32
            transform: 图片预处理变换，默认灰度化+Resize+ToTensor
        """
        self.root_dir = root_dir
        self.image_size = image_size

        # 默认预处理：灰度化 → 缩放到32×32 → 转Tensor（值归一化到[0,1]）
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        # 扫描所有图片文件
        self.image_paths = [
            os.path.join(root_dir, f)
            for f in sorted(os.listdir(root_dir))
            if f.lower().endswith(('.png', '.jpg', '.bmp'))
        ]
        assert len(self.image_paths) > 0, f"文件夹 {root_dir} 里没有图片"

    def __len__(self) -> int:
        """返回样本总数"""
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """
        加载第 index 张图片和标签
        返回: (image_tensor, label)
            image_tensor: shape=(1, 32, 32)，fp32精度
            label: 数字标签（0-9），从文件名提取
        """
        path = self.image_paths[index]
        image = Image.open(path).convert("L")  # 转灰度
        image = self.transform(image)           # 预处理 → Tensor
        image = image.float()                   # 确保 fp32 精度

        # 从文件名提取标签："3_045.png" → 3
        filename = os.path.basename(path)
        label = int(filename.split("_")[0])
        return image, label
