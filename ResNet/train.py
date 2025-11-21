import torch
import glob
import sys
import os
import wandb
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

from ResNet.model import Net
from colorize_data import ColorizeData
from metrics import compute_ssim, compute_deltaE 


class Trainer_ResNet:
    def __init__(self, train_paths, val_paths, latent_dim, lpips, epochs, batch_size, learning_rate, num_workers):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lpips = lpips

    def train(self):               
        train_dataset = ColorizeData(paths=self.train_paths)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=True)
        model = Net(256).to(self.device)
        criterion = torch.nn.MSELoss(reduction='mean').to(self.device)
        optimizer = torch.optim.Adam(model.parameters(),lr=self.learning_rate, weight_decay=1e-6)

        # train loop
        for epoch in range(self.epochs):
            print("Starting Training Epoch " + str(epoch + 1))
            avg_loss = 0.0
            model.train()
            for i, data in enumerate(tqdm(train_dataloader)):                                                    #(train_dataloader, 0)?
                inputs, targets = data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()                                                                       # Initialize optimizer 

                outputs = model(inputs)                                                                     # forward prop

                loss = torch.sqrt(criterion(outputs, targets))
                loss.backward()                                                                             # back prop
                optimizer.step()                                                                            # Update the weights

                avg_loss += loss.item()

            print(f'Epoch {epoch + 1} \t\t Training Loss: {avg_loss / len(train_dataloader)}')
            

            if (epoch + 1) % 1 == 0:
                val_metrics = self.validate(model, criterion)
                print(f"Epoch {epoch + 1} \t Validation Loss: {val_metrics["loss"]:.4f}")
                
            wandb.log({
                "train/epoch": epoch,
                "train/loss": avg_loss / len(train_dataloader),
                "val/loss": val_metrics["loss"],
                "val/ssim": val_metrics["ssim"],
                "val/lpips": val_metrics["lpips"],
                "val/deltaE": val_metrics["deltaE"],
            })
            torch.save(model.state_dict(), './Models/resnet/saved_model_' + str(epoch + 1) + '.pth')


    def validate(self, model, criterion):
        model.eval()

        total_loss = 0.0
        total_ssim = 0.0
        total_lpips = 0.0
        total_deltaE = 0.0
        count = 0

        with torch.no_grad():
            valid_loss = 0.0
            val_dataset = ColorizeData(paths=self.val_paths)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
            for i, data in enumerate(val_dataloader):
                inputs, targets = data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = model(inputs)

                loss = torch.sqrt(criterion(outputs, targets))
                
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