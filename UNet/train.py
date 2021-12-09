import torch
import glob

import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Net
from colorize_data import ColorizeData


class Trainer:
    def __init__(self, train_paths, val_paths, epochs, batch_size, learning_rate, num_workers):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
                val_loss, val_len = self.validate(model, criterion)
                print(f'Epoch {epoch + 1} \t\t Validation Loss: {val_loss / val_len}')
                
            torch.save(model.state_dict(), '../Results/RGB_ResNet/Models/saved_model_' + str(epoch + 1) + '.pth')


    def validate(self, model, criterion):
        # Validation loop begin
        # ------
        # Validation loop end
        # ------
        # Determine your evaluation metrics on the validation dataset.
        model.eval()
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

                valid_loss += loss.item()
        # model.train()
        return valid_loss, len(val_dataloader)


if __name__ == "__main__":
    path = "../Dataset/"
    paths = np.array(glob.glob(path + "/*.jpg"))
    rand_indices = np.random.permutation(len(paths))                                                                               # Number of images in dataset
    # I had reserved a few samples for testing on unseen data. These are now ignored.
    train_indices, val_indices, test_indices = rand_indices[:3600], rand_indices[3600:4000], rand_indices[4000:]
    train_paths = paths[train_indices]
    val_paths = paths[val_indices]

    trainer = Trainer(train_paths, val_paths, epochs = 200, batch_size = 64, learning_rate = 0.01, num_workers = 2)
    trainer.train()