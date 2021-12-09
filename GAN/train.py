import torch
import glob

import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

from model import NetGen, NetDis
from colorize_data import ColorizeData



class Trainer:
    def __init__(self, train_paths, val_paths, epochs, batch_size, learning_rate, num_workers):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.train_paths = train_paths
        self.val_paths = val_paths        
        self.real_label = 1
        self.fake_label = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
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
                errD_val, errG_val, val_len = self.validate(model_D, model_G, criterion, L1)
                print(f'Validation: Epoch {epoch + 1} \t\t Discriminator Loss: {errD_val / val_len}  \t\t Generator Loss: {errG_val / val_len}')
 
                
            torch.save(model_G.state_dict(), '../Results/RGB_GAN/Generator/saved_model_' + str(epoch + 1) + '.pth')
            torch.save(model_D.state_dict(), '../Results/RGB_GAN/Discriminator/saved_model_' + str(epoch + 1) + '.pth')


    def validate(self, model_D, model_G, criterion, L1):
        model_G.eval()
        model_D.eval()
        with torch.no_grad():
            valid_loss = 0.0
            val_dataset = ColorizeData(paths=self.val_paths)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last = True)
            for i, data in enumerate(val_dataloader):
                inputs, targets = data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                label = torch.full((self.batch_size,), self.real_label, dtype=torch.float, device=self.device)
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
                errG_L1 = L1(fake.view(fake.size(0),-1), targets.view(targets.size(0),-1))
                errG = errG + 100* errG_L1

        return errD, errG, len(val_dataloader)


if __name__ == "__main__":
    path = "../Dataset/"
    paths = np.array(glob.glob(path + "/*.jpg"))
    rand_indices = np.random.permutation(len(paths))                                                                               # Number of images in dataset
    # I had reserved a few samples for testing on unseen data. These are now ignored.
    train_indices, val_indices, test_indices = rand_indices[:3600], rand_indices[3600:4000], rand_indices[4000:]
    train_paths = paths[train_indices]
    val_paths = paths[val_indices]

    trainer = Trainer(train_paths, val_paths, epochs = 100, batch_size = 64, learning_rate = 0.01, num_workers = 2)
    trainer.train()