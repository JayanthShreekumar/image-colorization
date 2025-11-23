import wandb
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from colorize_data import ColorizeData
from UNetDiffusion.model import UNetDiffusion
from metrics import compute_ssim, compute_deltaE

def make_beta_schedule(T=1000, start=1e-4, end=0.02):
    return torch.linspace(start, end, T)


def diffusion_loss(model, x0, x_gray, betas, device):
    B = x0.size(0)
    T = betas.shape[0]

    t = torch.randint(0, T, (B,), device=device).long()

    alpha = 1.0 - betas
    alpha_bar = torch.cumprod(alpha, dim=0).to(device)

    noise = torch.randn_like(x0)

    sqrt_ab = alpha_bar[t].sqrt().reshape(B, 1, 1, 1)
    sqrt_mab = (1 - alpha_bar[t]).sqrt().reshape(B, 1, 1, 1)
    x_t = sqrt_ab * x0 + sqrt_mab * noise

    model_input = torch.cat([x_t, x_gray], dim=1)        # pred​(xt​,t ∣ gray) - conditioning on the grayscale image is essential

    noise_pred = model(model_input, t)                   # model outputs the prediction
    return F.mse_loss(noise_pred, noise)


class Trainer_DDPM:
    def __init__(self, train_paths, val_paths, latent_dim, device, lpips, epochs, batch_size, learning_rate, num_workers, **kwargs):

        self.train_paths = train_paths
        self.val_paths = val_paths
        self.model = UNetDiffusion(in_channels=4, out_channels=3, time_dim=latent_dim)
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers

        self.lpips = lpips

        # diffusion schedule
        self.T = 100
        self.betas = make_beta_schedule(self.T).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.timesteps = self.T

    def sample_ddpm(self, x_gray):
        self.model.eval()
        b = x_gray.size(0)
        device = x_gray.device

        x_t = torch.randn(b, 3, 256, 256).to(device)

        for t in reversed(range(self.T)):
            t_batch = torch.full((b,), t, dtype=torch.long, device=device)

            # Concatenate grayscale for conditioning
            model_input = torch.cat([x_t, x_gray], dim=1)

            eps_theta = self.model(model_input, t_batch)

            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            beta_t = self.betas[t]

            x0_pred = (x_t - (1 - alpha_t).sqrt() * eps_theta) / alpha_t.sqrt()

            mean = (
                alpha_bar_t.sqrt() * beta_t / (1 - alpha_bar_t) * x0_pred +
                (1 - alpha_t) / (1 - alpha_bar_t) * x_t
            )

            if t > 0:
                noise = torch.randn_like(x_t)
                sigma = beta_t.sqrt()
                x_t = mean + sigma * noise
            else:
                x_t = mean

        return torch.clamp(x_t, 0.0, 1.0)


    def train(self):
        train_dataset = ColorizeData(self.train_paths)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  pin_memory=True)

        val_dataset = ColorizeData(self.val_paths)
        val_loader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers,
                                pin_memory=True)

        self.model = self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            print(f"Starting Training Epoch {epoch + 1}")
            self.model.train()

            train_loss_sum = 0.0

            for x_gray, x_color in tqdm(train_loader):
                x_gray = x_gray.to(self.device)      # (B,1,H,W)
                x_color = x_color.to(self.device)    # (B,3,H,W)

                optimizer.zero_grad()

                loss = diffusion_loss(
                    model=self.model,
                    x0=x_color,
                    x_gray=x_gray,
                    betas=self.betas,
                    device=self.device
                )

                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()

            avg_train_loss = train_loss_sum / len(train_loader)
            print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

            # Validation metrics
            val_metrics = self.validate(val_loader)
            print(f"Epoch {epoch+1}  Validation Loss: {val_metrics['loss']:.4f}")

            wandb.log({
                "train/epoch": epoch,
                "train/loss": avg_train_loss,
                "val/loss": val_metrics["loss"],
                "val/ssim": val_metrics["ssim"],
                "val/lpips": val_metrics["lpips"],
                "val/deltaE": val_metrics["deltaE"],
            })

            # torch.save(self.model.state_dict(), f"./Models/unetddpm/saved_model_{epoch+1}.pth")


    def validate(self, val_loader):
        self.model.eval()

        total_loss = 0.0
        total_ssim = 0.0
        total_lpips = 0.0
        total_deltaE = 0.0
        count = 0

        with torch.no_grad():
            for x_gray, x_color in val_loader:
                x_gray  = x_gray.to(self.device)
                x_color = x_color.to(self.device)

                loss = diffusion_loss(
                    model=self.model,
                    x0=x_color,
                    x_gray=x_gray,
                    betas=self.betas,
                    device=self.device
                )
                total_loss += loss.item()

                recon = self.sample_ddpm(x_gray)

                total_ssim   += compute_ssim(recon, x_color)
                total_lpips  += self.lpips(recon, x_color).mean().item()
                total_deltaE += compute_deltaE(recon, x_color)

                count += 1

        return {
            "loss": total_loss / count,
            "ssim": total_ssim / count,
            "lpips": total_lpips / count,
            "deltaE": total_deltaE / count,
        }
    

class Trainer_DDIM:
    def __init__(self, train_paths, val_paths, latent_dim, device, lpips, epochs, batch_size, learning_rate, num_workers, **kwargs):

        self.train_paths = train_paths
        self.val_paths = val_paths
        self.model = UNetDiffusion(in_channels=4, out_channels=3, time_dim=latent_dim)
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.steps = kwargs.get("steps", 50)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers

        self.lpips = lpips

        # diffusion schedule
        self.T = 100
        self.betas = make_beta_schedule(self.T).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.timesteps = self.T

    def sample_ddim(self, x_gray, steps=50, eta=0.0):
        """
        DDIM sampling
        steps: number of sampling steps (<= self.T)
        eta: 0 for deterministic, >0 adds stochasticity
        """
        self.model.eval()
        device = x_gray.device
        B = x_gray.size(0)

        # precompute alphas and alpha_bars (already done in __init__)
        alpha_bars = self.alpha_bars
        alphas = self.alphas

        # select timesteps for DDIM
        ddim_steps = torch.linspace(0, self.T-1, steps, dtype=torch.long, device=device)

        # start from pure noise
        x_t = torch.randn(B, 3, 256, 256, device=device)

        for i in reversed(range(steps)):
            t = ddim_steps[i]
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            eps_theta = self.model(torch.cat([x_t, x_gray], dim=1), t_batch)

            alpha_bar_t = alpha_bars[t]
            alpha_bar_prev = alpha_bars[0] if i == 0 else alpha_bars[ddim_steps[i-1]]

            # DDIM update
            x0_pred = (x_t - (1 - alpha_bar_t).sqrt() * eps_theta) / (alpha_bar_t.sqrt())
            sigma_t = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)).sqrt()

            noise = torch.randn_like(x_t) if i > 0 else 0.0
            x_t = alpha_bar_prev.sqrt() * x0_pred + (1 - alpha_bar_prev - sigma_t**2).sqrt() * eps_theta + sigma_t * noise

        return torch.clamp(x_t, 0.0, 1.0)


    def train(self):
        train_dataset = ColorizeData(self.train_paths)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  pin_memory=True)

        val_dataset = ColorizeData(self.val_paths)
        val_loader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers,
                                pin_memory=True)

        self.model = self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            print(f"Starting Training Epoch {epoch + 1}")
            self.model.train()

            train_loss_sum = 0.0

            for x_gray, x_color in tqdm(train_loader):
                x_gray = x_gray.to(self.device)      # (B,1,H,W)
                x_color = x_color.to(self.device)    # (B,3,H,W)

                optimizer.zero_grad()

                loss = diffusion_loss(
                    model=self.model,
                    x0=x_color,
                    x_gray=x_gray,
                    betas=self.betas,
                    device=self.device
                )

                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()

            avg_train_loss = train_loss_sum / len(train_loader)
            print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

            # Validation metrics
            val_metrics = self.validate(val_loader, steps=self.steps)
            print(f"Epoch {epoch+1}  Validation Loss: {val_metrics['loss']:.4f}")

            wandb.log({
                "train/epoch": epoch,
                "train/loss": avg_train_loss,
                "val/loss": val_metrics["loss"],
                "val/ssim": val_metrics["ssim"],
                "val/lpips": val_metrics["lpips"],
                "val/deltaE": val_metrics["deltaE"],
            })

            # torch.save(self.model.state_dict(), f"./Models/unetddim/saved_model_{epoch+1}.pth")


    def validate(self, val_loader, steps=50):
        self.model.eval()

        total_loss = 0.0
        total_ssim = 0.0
        total_lpips = 0.0
        total_deltaE = 0.0
        count = 0

        with torch.no_grad():
            for x_gray, x_color in val_loader:
                x_gray  = x_gray.to(self.device)
                x_color = x_color.to(self.device)

                loss = diffusion_loss(
                    model=self.model,
                    x0=x_color,
                    x_gray=x_gray,
                    betas=self.betas,
                    device=self.device
                )
                total_loss += loss.item()

                recon = self.sample_ddim(x_gray, steps=steps)

                total_ssim   += compute_ssim(recon, x_color)
                total_lpips  += self.lpips(recon, x_color).mean().item()
                total_deltaE += compute_deltaE(recon, x_color)

                count += 1

        return {
            "loss": total_loss / count,
            "ssim": total_ssim / count,
            "lpips": total_lpips / count,
            "deltaE": total_deltaE / count,
        }
    

