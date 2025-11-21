import torch
import glob
import sys
import os
import wandb
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

from UNet_GAN.model import NetGen, NetDis
from colorize_data import ColorizeData
from metrics import compute_ssim, compute_deltaE


class Trainer_UNet_GAN:
    def __init__(self, train_paths, val_paths, latent_dim, lpips, epochs, batch_size, learning_rate, num_workers):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.real_label = 1
        self.fake_label = 0
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.lpips = lpips
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def train(self):               
        train_dataset = ColorizeData(paths=self.train_paths)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=True, drop_last = True)
        # Model
        model_G = NetGen().to(self.device)
        model_D = NetDis().to(self.device)
        model_G.apply(self.weights_init)
        model_D.apply(self.weights_init)

        optimizer_G = torch.optim.Adam(model_G.parameters(),
                             lr=self.learning_rate, betas=(0.5, 0.999),
                             eps=1e-8, weight_decay=0)
        optimizer_D = torch.optim.Adam(model_D.parameters(),
                             lr=self.learning_rate, betas=(0.5, 0.999),
                             eps=1e-8, weight_decay=0)
        
        criterion = nn.BCELoss()
        L1 = nn.L1Loss()

        model_G.train()
        model_D.train()


        # train loop
        for epoch in range(self.epochs):
            print("Starting Training Epoch " + str(epoch + 1))
            for i, data in enumerate(tqdm(train_dataloader)):                                                    #(train_dataloader, 0)?
                inputs, targets = data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                model_D.zero_grad()
                label = torch.full((self.batch_size,), self.real_label, dtype=torch.float, device=self.device)
                output = model_D(targets)
                errD_real = criterion(torch.squeeze(output), label)
                errD_real.backward()

                fake = model_G(inputs)
                label.fill_(self.fake_label)
                output = model_D(fake.detach())
                errD_fake = criterion(torch.squeeze(output), label)
                errD_fake.backward()
                errD = errD_real + errD_fake
                optimizer_D.step()

                model_G.zero_grad()
                label.fill_(self.real_label)
                output = model_D(fake)
                errG = criterion(torch.squeeze(output), label)
                errG_L1 = L1(fake.view(fake.size(0),-1), targets.view(targets.size(0),-1))
                errG = errG + 100 * errG_L1
                errG.backward()
                optimizer_G.step()   

            print(f'Training: Epoch {epoch + 1} \t\t Discriminator Loss: {errD / len(train_dataloader)}  \t\t Generator Loss: {errG / len(train_dataloader)}')
            
            if (epoch + 1) % 1 == 0:
                errD_val, errG_val, ssim_val, lpips_val, deltaE_val, val_len = self.validate(model_D, model_G, criterion, L1)
                print(f'Validation: Epoch {epoch + 1} \t\t Discriminator Loss: {errD_val / val_len}  \t\t Generator Loss: {errG_val / val_len}')

            wandb.log({
                        "train/epoch": epoch + 1,
                        "train/disc_loss": errD / len(train_dataloader),
                        "train/loss": errD / len(train_dataloader),
                        "val/disc_loss": errD_val,
                        "val/loss": errG_val,
                        "val/ssim": ssim_val,
                        "val/lpips": lpips_val,
                        "val/deltaE": deltaE_val
                    })
            torch.save(model_G.state_dict(), f'./Models/unetgan/saved_model_' + str(epoch + 1) + '.pth')
            # torch.save(model_D.state_dict(), f'./Results/RGB_GAN/Discriminator/saved_model_' + str(epoch + 1) + '.pth')


    
    def validate(self, model_D, model_G, criterion, L1):
        model_G.eval()
        model_D.eval()

        total_errD = 0.0
        total_errG = 0.0
        total_ssim = 0.0
        total_lpips = 0.0
        total_deltaE = 0.0
        count = 0

        val_dataset = ColorizeData(paths=self.val_paths)
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=True, drop_last=True
        )

        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                batch_size = inputs.size(0)
                count += batch_size

                label = torch.full((batch_size,), self.real_label, dtype=torch.float, device=self.device)
                output = model_D(targets)
                errD_real = criterion(torch.squeeze(output), label)

                fake = model_G(inputs)
                label.fill_(self.fake_label)
                output = model_D(fake.detach())
                errD_fake = criterion(torch.squeeze(output), label)
                errD = errD_real + errD_fake

                label.fill_(self.real_label)
                output = model_D(fake)
                errG = criterion(torch.squeeze(output), label)
                errG_L1 = L1(fake.view(batch_size, -1), targets.view(batch_size, -1))
                errG = errG + 100 * errG_L1

                total_errD += errD.item()
                total_errG += errG.item()

                total_ssim += compute_ssim(fake, targets).item() * batch_size
                total_lpips += self.lpips(fake, targets).item() * batch_size
                total_deltaE += compute_deltaE(fake, targets) * batch_size

        avg_errD = total_errD / len(val_dataloader)
        avg_errG = total_errG / len(val_dataloader)
        avg_ssim = total_ssim / count
        avg_lpips = total_lpips / count
        avg_deltaE = total_deltaE / count

        return avg_errD, avg_errG, avg_ssim, avg_lpips, avg_deltaE, len(val_dataloader)