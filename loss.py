import torch
from torch import nn
from torchvision.models import vgg19


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        model = vgg19(pretrained=True)
        self.vgg = model.features[:18].eval()  # Use the features up to the 18th layer
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input_vgg = self.vgg(input)
        target_vgg = self.vgg(target)
        loss = nn.functional.mse_loss(input_vgg, target_vgg)
        return loss


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.vgg_loss = VGGLoss()
        self.adversarial_loss = nn.BCEWithLogitsLoss()

    def forward(self, fake_amp_pred, fake_phase_pred, real_amp_pred, real_phase_pred, out_amp_images, target_amp_images,
                out_phase_images, target_phase_images):
        # Adversarial Loss for amplitude and phase
        real_amp_pred_mean = torch.mean(real_amp_pred.detach())
        real_phase_pred_mean = torch.mean(real_phase_pred.detach())

        # Adversarial Loss for amplitude and phase
        adversarial_loss_amp = self.adversarial_loss(fake_amp_pred - real_amp_pred_mean,
                                                     torch.ones_like(fake_amp_pred))
        adversarial_loss_phase = self.adversarial_loss(fake_phase_pred - real_phase_pred_mean,
                                                       torch.ones_like(fake_phase_pred))

        # Pixel-wise Loss
        l1_loss_amp = self.l1_loss(out_amp_images, target_amp_images)
        l1_loss_phase = self.l1_loss(out_phase_images, target_phase_images)

        #VGG Loss for amplitude and phase
        vgg_loss_amp = self.vgg_loss(out_amp_images, target_amp_images)
        vgg_loss_phase = self.vgg_loss(out_phase_images, target_phase_images)

        total_loss = (l1_loss_amp + l1_loss_phase + 0.001 * (
                    adversarial_loss_amp + adversarial_loss_phase)+vgg_loss_amp+vgg_loss_phase)
        return total_loss


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, real_amp_pred, fake_amp_pred, real_phase_pred, fake_phase_pred):
        # Adversarial loss for real amplitude images (RaGAN)
        real_loss_amp = self.criterion(real_amp_pred - fake_amp_pred.mean(0, keepdim=True),
                                       torch.ones_like(real_amp_pred))
        real_loss_amp = torch.mean(real_loss_amp)

        # Adversarial loss for fake amplitude images (RaGAN)
        fake_loss_amp = self.criterion(fake_amp_pred - real_amp_pred.mean(0, keepdim=True),
                                       torch.zeros_like(fake_amp_pred))
        fake_loss_amp = torch.mean(fake_loss_amp)

        # Adversarial loss for real phase images (RaGAN)
        real_loss_phase = self.criterion(real_phase_pred - fake_phase_pred.mean(0, keepdim=True),
                                         torch.ones_like(real_phase_pred))
        real_loss_phase = torch.mean(real_loss_phase)

        # Adversarial loss for fake phase images (RaGAN)
        fake_loss_phase = self.criterion(fake_phase_pred - real_phase_pred.mean(0, keepdim=True),
                                         torch.zeros_like(fake_phase_pred))
        fake_loss_phase = torch.mean(fake_loss_phase)

        # Total discriminator loss
        d_loss = (real_loss_amp + fake_loss_amp + real_loss_phase + fake_loss_phase) / 4
        return d_loss