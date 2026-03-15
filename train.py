import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Subset

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs("gan_output", exist_ok=True)
print("Output directory 'gan_output/' ready.")

# ── Hyperparameters ───────────────────────────────────────────────────────────
Z_DIM        = 100
LR           = 0.0002
BETAS        = (0.5, 0.999)
BATCH_SIZE   = 64
EPOCHS       = 50
LABEL_SMOOTH = 0.9
IMAGE_SIZE   = 64
NC           = 3
NGF          = 64
NDF          = 64
SAVE_EVERY   = 5

print(f"Hyperparameters: Z_DIM={Z_DIM}, LR={LR}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}")

# ── Dataset ───────────────────────────────────────────────────────────────────
DOG_CLASS = 5

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

full_train = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=transform)
full_test  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

dog_train_idx = [i for i, (_, lbl) in enumerate(full_train) if lbl == DOG_CLASS]
dog_test_idx  = [i for i, (_, lbl) in enumerate(full_test)  if lbl == DOG_CLASS]

dog_dataset = torch.utils.data.ConcatDataset([
    Subset(full_train, dog_train_idx),
    Subset(full_test,  dog_test_idx),
])

dataloader = DataLoader(
    dog_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=(device.type == "cuda"),
    drop_last=True,
)

print(f"Dog images found: {len(dog_dataset)}")
print(f"Batches per epoch: {len(dataloader)}")

# ── Generator ─────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM, ngf=NGF, nc=NC):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)

# ── Discriminator ─────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    def __init__(self, ndf=NDF, nc=NC):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, img):
        return self.net(img).view(-1)

# ── Weights Init ──────────────────────────────────────────────────────────────
def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(m.bias.data, 0.0)

netG = Generator().to(device)
netD = Discriminator().to(device)
netG.apply(weights_init)
netD.apply(weights_init)

# ── Loss & Optimizers ─────────────────────────────────────────────────────────
criterion  = nn.BCELoss()
REAL_LABEL = LABEL_SMOOTH
FAKE_LABEL = 0.0
optimizerD = torch.optim.Adam(netD.parameters(), lr=LR, betas=BETAS)
optimizerG = torch.optim.Adam(netG.parameters(), lr=LR, betas=BETAS)
fixed_noise = torch.randn(64, Z_DIM, 1, 1, device=device)

# ── Training Loop ─────────────────────────────────────────────────────────────
G_losses, D_losses = [], []

print("Starting training...")
print(f"{'Epoch':>6} | {'D Loss':>10} | {'G Loss':>10}")
print("-" * 34)

for epoch in range(1, EPOCHS + 1):
    epoch_d_loss = epoch_g_loss = 0.0
    num_batches = 0

    netG.train()
    netD.train()

    for real_imgs, _ in dataloader:
        batch_size = real_imgs.size(0)
        real_imgs  = real_imgs.to(device)

        real_labels = torch.full((batch_size,), REAL_LABEL, device=device)
        fake_labels = torch.full((batch_size,), FAKE_LABEL,  device=device)

        # Update Discriminator
        netD.zero_grad()
        output_real = netD(real_imgs)
        loss_D_real = criterion(output_real, real_labels)
        loss_D_real.backward()

        noise     = torch.randn(batch_size, Z_DIM, 1, 1, device=device)
        fake_imgs = netG(noise)
        output_fake = netD(fake_imgs.detach())
        loss_D_fake = criterion(output_fake, fake_labels)
        loss_D_fake.backward()
        loss_D = loss_D_real + loss_D_fake
        optimizerD.step()

        # Update Generator
        netG.zero_grad()
        output_fake_for_G = netD(fake_imgs)
        loss_G = criterion(output_fake_for_G, real_labels)
        loss_G.backward()
        optimizerG.step()

        epoch_d_loss += loss_D.item()
        epoch_g_loss += loss_G.item()
        num_batches  += 1

    avg_d = epoch_d_loss / num_batches
    avg_g = epoch_g_loss / num_batches
    D_losses.append(avg_d)
    G_losses.append(avg_g)
    print(f"{epoch:>6} | {avg_d:>10.4f} | {avg_g:>10.4f}")

    if epoch % SAVE_EVERY == 0:
        netG.eval()
        with torch.no_grad():
            fake_sample = netG(fixed_noise).cpu()
        save_path = f"gan_output/epoch_{epoch:04d}.png"
        vutils.save_image(fake_sample, save_path, nrow=8, normalize=True, value_range=(-1, 1))
        print(f"  -> Saved sample grid to {save_path}")

print("\nTraining complete!")

# ── Loss Curves ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(1, len(G_losses) + 1), D_losses, label="Discriminator Loss", color="royalblue", linewidth=2)
ax.plot(range(1, len(G_losses) + 1), G_losses, label="Generator Loss",     color="tomato",    linewidth=2)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.set_title("GAN Training Losses", fontweight="bold")
ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("gan_output/loss_curves.png", dpi=150)
print("Loss curve saved to gan_output/loss_curves.png")
