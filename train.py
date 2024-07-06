import os
from math import log10
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import pytorch_ssim
from dataloader import TrainDatasetFromFolder, ValDatasetFromFolder
from loss import GeneratorLoss, DiscriminatorLoss
from model import Generator, Discriminator
from torch.nn import BCEWithLogitsLoss
from spikingjelly.activation_based import functional

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#启用异常检测
torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    CROP_SIZE = 256
    NUM_EPOCHS = 50
    BATCH_SIZE = 8  # 减少批量大小
    NUM_WORKERS = 4
    COLOR_MODE = 'RGB'  # 使用 RGB 模式

    # 创建数据加载器, 使用 pin_memory=True
    train_set = TrainDatasetFromFolder('dataset_restore', crop_size=CROP_SIZE, color_mode=COLOR_MODE)
    train_loader = DataLoader(dataset=train_set, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_set = ValDatasetFromFolder('val_images', crop_size=CROP_SIZE, color_mode=COLOR_MODE)
    val_loader = DataLoader(dataset=val_set, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # 初始化生成器和判别器
    netG = Generator(color_mode=COLOR_MODE)
    netD = Discriminator(color_mode=COLOR_MODE)

    # 检查 CUDA 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG.to(device)
    netD.to(device)

    # 定义损失函数和优化器
    generator_criterion = GeneratorLoss().to(device)
    discriminator_criterion = DiscriminatorLoss().to(device)
    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters(), lr=0.001)

    # 创建 TensorBoard 写入器
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join('board', current_time)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        for target_phase, target_amp, data in train_bar:
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            # 将输入和目标数据加载到 GPU 上
            real_phase_img = target_phase.to(device, non_blocking=True)
            real_amp_img = target_amp.to(device, non_blocking=True)
            noisy_img = data.to(device, non_blocking=True)
            fake_phase_img, fake_amp_img = netG(noisy_img)

            # 更新判别器
            optimizerD.zero_grad()
            real_phase_pred = netD(real_phase_img)
            fake_phase_pred = netD(fake_phase_img.detach())
            real_amp_pred = netD(real_amp_img)
            fake_amp_pred = netD(fake_amp_img.detach())

            d_loss = discriminator_criterion(real_amp_pred, fake_amp_pred, real_phase_pred, fake_phase_pred)
            d_loss.backward(retain_graph=True)
            optimizerD.step()
            # functional.reset_net(netD)

            # 更新生成器
            optimizerG.zero_grad()
            fake_phase_pred_2 = netD(fake_phase_img)
            fake_amp_pred_2 = netD(fake_amp_img)
            g_loss = generator_criterion(fake_amp_pred_2, fake_phase_pred_2, real_amp_pred, real_phase_pred, fake_amp_img, real_amp_img, fake_phase_img, real_phase_img)
            g_loss.backward()
            optimizerG.step()
            # functional.reset_net(netG)

            # 记录当前批次的损失和分数
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += (real_phase_pred.mean().item() + real_amp_pred.mean().item()) * batch_size / 2
            running_results['g_score'] += (fake_phase_pred.mean().item() + fake_amp_pred.mean().item()) * batch_size / 2

            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        # 在每个 epoch 结束时, 使用验证集图像生成图像并保存到 TensorBoard
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            val_images = []
            val_targets_phase = []
            val_targets_amp = []
            for val_target_phase, val_target_amp, val_data in val_bar:
                val_images.append(val_data.to(device, non_blocking=True))
                val_targets_phase.append(val_target_phase.to(device, non_blocking=True))
                val_targets_amp.append(val_target_amp.to(device, non_blocking=True))
            val_images = torch.cat(val_images, dim=0)
            val_targets_phase = torch.cat(val_targets_phase, dim=0)
            val_targets_amp = torch.cat(val_targets_amp, dim=0)
            fake_val_phase_images, fake_val_amp_images = netG(val_images).detach().cpu()
            val_images = val_images.cpu()
            img_grid_fake_phase = utils.make_grid(fake_val_phase_images, normalize=True)
            img_grid_fake_amp = utils.make_grid(fake_val_amp_images, normalize=True)
            img_grid_real_phase = utils.make_grid(val_targets_phase.cpu(), normalize=True)
            img_grid_real_amp = utils.make_grid(val_targets_amp.cpu(), normalize=True)
            writer.add_image(f'Fake Phase Images', img_grid_fake_phase, global_step=epoch)
            writer.add_image(f'Fake Amplitude Images', img_grid_fake_amp, global_step=epoch)
            writer.add_image(f'Real Phase Images', img_grid_real_phase, global_step=epoch)
            writer.add_image(f'Real Amplitude Images', img_grid_real_amp, global_step=epoch)

        # 保存模型参数
        torch.save(netG.state_dict(), f'checkpoints/netG_epoch_{epoch}.pth')
        torch.save(netD.state_dict(), f'checkpoints/netD_epoch_{epoch}.pth')

        # 将每个 epoch 的损失和分数写入 TensorBoard
        writer.add_scalar('Loss/Discriminator', running_results['d_loss'] / running_results['batch_sizes'], epoch)
        writer.add_scalar('Loss/Generator', running_results['g_loss'] / running_results['batch_sizes'], epoch)
        writer.add_scalar('Score/Real', running_results['d_score'] / running_results['batch_sizes'], epoch)
        writer.add_scalar('Score/Fake', running_results['g_score'] / running_results['batch_sizes'], epoch)

        # 清空 GPU 缓存
        torch.cuda.empty_cache()
    writer.close()
