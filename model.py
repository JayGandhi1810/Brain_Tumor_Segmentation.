"""
U-Net Model Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


# Metrics

def dice_coefficient(pred, target, smooth=1e-6):
    pred   = (pred > 0.5).float()
    inter  = (pred * target).sum()
    return (2. * inter + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1e-6):
    pred   = (pred > 0.5).float()
    inter  = (pred * target).sum()
    union  = pred.sum() + target.sum() - inter
    return (inter + smooth) / (union + smooth)


# Loss

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce    = nn.BCELoss()

    def forward(self, pred, target):
        # BCE
        bce_loss = self.bce(pred, target)

        # Dice
        pred_f   = pred.view(-1)
        tgt_f    = target.view(-1)
        inter    = (pred_f * tgt_f).sum()
        dice_loss = 1 - (2. * inter + self.smooth) / \
                        (pred_f.sum() + tgt_f.sum() + self.smooth)

        return bce_loss + dice_loss


# Building blocks

class ConvBlock(nn.Module):
    """Two Conv2d → BN → ReLU layers"""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.block(x))


class EncoderBlock(nn.Module):
    """ConvBlock + MaxPool; returns (skip, pooled)"""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, dropout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        return skip, self.pool(skip)


class DecoderBlock(nn.Module):
    """ConvTranspose + skip concat + ConvBlock"""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# U-Net

class UNet(nn.Module):
    def __init__(self,
                 in_channels = config.IMG_CHANNELS,
                 filters     = config.FILTERS,
                 dropout     = config.DROPOUT_RATE):
        super().__init__()

        # Encoder
        self.encoders = nn.ModuleList()
        ch = in_channels
        for f in filters[:-1]:
            self.encoders.append(EncoderBlock(ch, f, dropout))
            ch = f

        # Bottleneck
        self.bottleneck = ConvBlock(ch, filters[-1], dropout)
        ch = filters[-1]

        # Decoder
        self.decoders = nn.ModuleList()
        for f in reversed(filters[:-1]):
            self.decoders.append(DecoderBlock(ch, f, dropout))
            ch = f

        # Output
        self.output_conv = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        skips = []

        # Encode
        for enc in self.encoders:
            skip, x = enc(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decode
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return torch.sigmoid(self.output_conv(x))


# Factory

def get_model(device):
    """Create model, loss, and optimizer"""
    model     = UNet().to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max',
        factor  = config.REDUCE_LR_FACTOR,
        patience= config.REDUCE_LR_PATIENCE,
        min_lr  = config.MIN_LR)

    # Count parameters
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model ready | Trainable parameters: {total:,}")

    return model, criterion, optimizer, scheduler


def load_model(path, device):
    model = UNet().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Model loaded from: {path}")
    return model
