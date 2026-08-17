import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. Architecture (Clean U-Net SR)
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)


class UNetSR(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for feature in reversed(features):
            self.ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(feature * 2, feature, kernel_size=3, padding=1)
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final 2x Super-Resolution Head
        self.upscale_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(features[0], features[0], kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(features[0], out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            concat_x = torch.cat((skip_connections[idx // 2], x), dim=1)
            x = self.ups[idx + 1](concat_x)

        return self.upscale_head(x)

# ==========================================
# 2. Smooth & Sharp Combined Loss
# ==========================================
class SmoothSharpLoss(nn.Module):
    def __init__(self, eps=1e-3, tv_weight=0.05, edge_weight=0.4):
        super().__init__()
        self.eps = eps
        self.tv_weight = tv_weight
        self.edge_weight = edge_weight

    def total_variation_loss(self, img):
        # Penalizes high-frequency noise spikes in flat/textured areas
        tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
        return tv_h + tv_w

    def forward(self, pred, gt):
        # 1. Base Charbonnier Reconstruction Loss
        diff = pred - gt
        loss_charbonnier = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))

        # 2. Moderated Gradient Difference (Controlled Edge Retention)
        pred_grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        gt_grad_x = torch.abs(gt[:, :, :, :-1] - gt[:, :, :, 1:])
        pred_grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        gt_grad_y = torch.abs(gt[:, :, :-1, :] - gt[:, :, 1:, :])

        edge_loss = torch.mean(torch.abs(pred_grad_x - gt_grad_x)) + \
                    torch.mean(torch.abs(pred_grad_y - gt_grad_y))

        # 3. Total Variation Loss (Grain Suppression)
        tv_loss = self.total_variation_loss(pred)

        return loss_charbonnier + (self.edge_weight * edge_loss) + (self.tv_weight * tv_loss)

# ==========================================
# 3. Dataset Handling with Augmentations
# ==========================================
class SuperResDataset(Dataset):
    def __init__(self, lr_dir, gt_dir, augment=True):
        self.lr_dir = lr_dir
        self.gt_dir = gt_dir
        self.filenames = sorted([f for f in os.listdir(lr_dir) if f.endswith('.npy')])
        self.augment = augment

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        lr_np = np.clip(np.load(os.path.join(self.lr_dir, filename)).astype(np.float32), 0.0, 1.0)
        gt_np = np.clip(np.load(os.path.join(self.gt_dir, filename)).astype(np.float32), 0.0, 1.0)

        if self.augment:
            if random.random() > 0.5:
                lr_np = np.fliplr(lr_np).copy()
                gt_np = np.fliplr(gt_np).copy()

            if random.random() > 0.5:
                lr_np = np.flipud(lr_np).copy()
                gt_np = np.flipud(gt_np).copy()

            rot_k = random.choice([0, 1, 2, 3])
            if rot_k > 0:
                lr_np = np.rot90(lr_np, k=rot_k).copy()
                gt_np = np.rot90(gt_np, k=rot_k).copy()

        lr_img = torch.from_numpy(lr_np).unsqueeze(0) if lr_np.ndim == 2 else torch.from_numpy(lr_np)
        gt_img = torch.from_numpy(gt_np).unsqueeze(0) if gt_np.ndim == 2 else torch.from_numpy(gt_np)

        return lr_img, gt_img

# ==========================================
# 4. Training Loop
# ==========================================
def train():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16  
    TOTAL_EPOCHS = 30
    START_EPOCH = 0

    LR_DIR = os.path.join(SCRIPT_DIR, "train", "train", "NoisyLR")
    GT_DIR = os.path.join(SCRIPT_DIR, "train", "train", "GT")
    WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "unet_sr_model_v4.pth")

    dataset = SuperResDataset(lr_dir=LR_DIR, gt_dir=GT_DIR, augment=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    model = UNetSR(in_channels=1, out_channels=1).to(DEVICE)

    if os.path.exists(WEIGHTS_PATH):
        print(f"Found checkpoint at '{WEIGHTS_PATH}'. Loading weights...")
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    else:
        print(f"Starting fresh training for {TOTAL_EPOCHS} epochs...")

    # Updated Loss Function with lower edge weight + Total Variation smoothing
    criterion = SmoothSharpLoss(edge_weight=0.4, tv_weight=0.05).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)

    scaler = torch.amp.GradScaler('cuda') if DEVICE == "cuda" else None
    torch.backends.cudnn.benchmark = True

    print(f"Training on {DEVICE}...")

    model.train()
    for epoch in range(START_EPOCH, TOTAL_EPOCHS):
        epoch_loss = 0.0
        for step, (noisy_lr, clean_gt) in enumerate(loader):
            noisy_lr, clean_gt = noisy_lr.to(DEVICE), clean_gt.to(DEVICE)

            optimizer.zero_grad()
            if DEVICE == "cuda":
                with torch.amp.autocast('cuda'):
                    predictions = model(noisy_lr)
                    loss = criterion(predictions, clean_gt)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = model(noisy_lr)
                loss = criterion(predictions, clean_gt)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        current_lr = scheduler.get_last_lr()[0]

        torch.save(model.state_dict(), WEIGHTS_PATH)
        print(f"Epoch [{epoch+1:02d}/{TOTAL_EPOCHS}] Completed | Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | Saved.")

    print(f"\nTraining complete! Saved weights to '{WEIGHTS_PATH}'")

if __name__ == "__main__":
    train()