import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor, CenterCrop, Resize, Compose, Pad
import torchvision.utils as utils

from model import Generator

def test_model(model_path, test_image_path, output_path):
    # 加载训练好的生成器模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(color_mode="L").to(device)
    generator.load_state_dict(torch.load(model_path))
    generator.eval()

    # 读取测试图像并进行预处理
    test_image = Image.open(test_image_path).convert('L')
    original_size = test_image.size

    # 计算填充大小
    pad_height = (32 - (original_size[0] % 32)) % 32
    pad_width = (32 - (original_size[1] % 32)) % 32
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    print(pad_top, pad_bottom, pad_left, pad_right)

    # 定义测试图像的预处理操作
    transform = Compose([
        Pad((pad_top, pad_bottom, pad_left, pad_right)),
        ToTensor()
    ])

    print(f"Original size: {original_size}")
    test_image = transform(test_image).unsqueeze(0).to(device)
    print(f"Padded size: {test_image.shape}")
    print(test_image.shape)
    # 使用训练好的生成器模型进行图像去噪
    with torch.no_grad():
        output_image = generator(test_image)

    # 后处理并保存去噪后的图像
    utils.save_image(output_image, output_path)


if __name__ == "__main__":
    model_path = "checkpoints/netG_epoch_100.pth"  # 训练好的生成器模型路径
    test_image_path = "test_images/101.jpeg"  # 测试图像路径
    output_path = ("results/img.png")  # 输出重建图像路径

    # 创建输出目录
    os.makedirs("results", exist_ok=True)

    # 测试模型
    test_model(model_path, test_image_path, output_path)