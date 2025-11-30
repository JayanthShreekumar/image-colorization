import torch
import glob
import sys
import os
import wandb
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm

from BasicVAE.model import BasicVAE
from colorize_data import ColorizeData
from metrics import compute_ssim, compute_deltaE 

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')  # instead of sum
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.001 * kl_loss

class Trainer_BasicVAE:
    def __init__(self, train_paths, val_paths, latent_dim, device, lpips, epochs=50, batch_size=8, learning_rate=1e-3, num_workers=4, **kwargs):
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.lpips = lpips

    def train(self):
        train_dataset = ColorizeData(paths=self.train_paths)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, pin_memory=True)

        val_dataset = ColorizeData(paths=self.val_paths)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=True)

        model = BasicVAE(in_channels=1, out_channels=3, latent_dim=self.latent_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-6)

        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0

            print(f"Starting Training Epoch {epoch + 1}")
            for inputs, targets in tqdm(train_loader):
                inputs = inputs.to(self.device)    # grayscale images
                targets = targets.to(self.device)  # RGB images

                optimizer.zero_grad()
                outputs, mu, logvar = model(inputs)
                loss = vae_loss(outputs, targets, mu, logvar)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch + 1} \t Training Loss: {avg_train_loss:.4f}")

            val_metrics = self.validate(model, val_loader)
            print(f"Epoch {epoch + 1} \t Validation Loss: {val_metrics["loss"]:.4f}")

            wandb.log({
                "train/epoch": epoch,
                "train/loss": avg_train_loss,
                "val/loss": val_metrics["loss"],
                "val/ssim": val_metrics["ssim"],
                "val/lpips": val_metrics["lpips"],
                "val/deltaE": val_metrics["deltaE"],
            })

        torch.save(model.state_dict(), f'./Models/basicvae/saved_model_{epoch+1}.pth')

    def validate(self, model, val_loader):
        model.eval()

        total_loss = 0.0
        total_ssim = 0.0
        total_lpips = 0.0
        total_deltaE = 0.0
        count = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs, mu, logvar = model(inputs)
                loss = vae_loss(outputs, targets, mu, logvar)

                # Metrics
                ssim_score = compute_ssim(outputs, targets)
                lpips_score = self.lpips(outputs, targets).mean().item()
                deltaE_score = compute_deltaE(outputs, targets)

                total_loss += loss.item()
                total_ssim += ssim_score
                total_lpips += lpips_score
                total_deltaE += deltaE_score
                count += 1

        return {
            "loss": total_loss / count,
            "ssim": total_ssim / count,
            "lpips": total_lpips / count,
            "deltaE": total_deltaE / count,
        }