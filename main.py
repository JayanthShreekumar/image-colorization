import glob
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from GAN.train import Trainer_GAN
from UNet.train import Trainer_UNet
from ResNet.train import Trainer_ResNet
from UNet_GAN.train import Trainer_UNet_GAN
from BasicVAE.train import Trainer_BasicVAE
from UNetVAE.train import Trainer_UNetVAE

from GAN.model import NetGen as GANColorizer
from UNet.model import Net as UNetColorizer
from ResNet.model import Net as ResNetColorizer
from UNet_GAN.model import NetGen as UNetGANColorizer
from BasicVAE.model import BasicVAE as BasicVAEColorizer
from UNetVAE.model import UNetVAE as UNetVAEColorizer

from colorize_data import ColorizeData
from metrics import compute_ssim, compute_deltaE, LPIPSWrapper

import wandb
wandb.login()

TRAINER_MAP = {
    "resnet": Trainer_ResNet,
    "unet": Trainer_UNet,
    "gan": Trainer_GAN,
    "unetgan": Trainer_UNet_GAN,
    "basicvae": Trainer_BasicVAE,
    "unetvae": Trainer_UNetVAE,
}

MODEL_MAP = {
    "resnet": ResNetColorizer,
    "unet": UNetColorizer,
    "gan": GANColorizer,
    "unetgan": UNetGANColorizer,
    "basicvae": BasicVAEColorizer,
    "unetvae": UNetVAEColorizer,
}

def create_dirs():
    os.makedirs("./Models/unet", exist_ok=True)
    os.makedirs("./Models/unetgan", exist_ok=True)
    os.makedirs("./Models/resnet", exist_ok=True)
    os.makedirs("./Models/gan", exist_ok=True)
    os.makedirs("./Models/basicvae", exist_ok=True)
    os.makedirs("./Models/unetvae", exist_ok=True)

def test(model_name, weights_path):
    print(f"\n Testing {model_name} weights at epoch {weights_path}\n")
    
    # Load model
    model = MODEL_MAP[model_name]()
    model.load_state_dict(
        torch.load(f"./Models/{model_name}/saved_model_{weights_path}.pth", map_location="cpu")
    )
    model.eval()

    # Load test images
    test_dir = "./TestImages/"
    test_paths = [os.path.join(test_dir, x) for x in os.listdir(test_dir)]

    test_dataset = ColorizeData(paths=test_paths)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    gray_batch, real_batch = next(iter(test_dataloader))  # one batch of 4 images
    preds = model(gray_batch)                             # forward pass

    if model_name in ["basicvae", "unetvae"]:
        preds = preds[0]
    # Plotting
    fig, ax = plt.subplots(4, 3, figsize=(12, 16))

    for i in range(4):
        # Convert to numpy and transpose
        gray_np = gray_batch[i].cpu().numpy().transpose(1,2,0)
        pred_np = preds[i].detach().cpu().numpy().transpose(1,2,0)
        real_np = real_batch[i].cpu().numpy().transpose(1,2,0)

        # Normalize predicted and real images to [0,1] for display
        pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
        real_np = (real_np - real_np.min()) / (real_np.max() - real_np.min() + 1e-8)

        # Display
        ax[i, 0].imshow(np.squeeze(gray_np), cmap="gray")
        ax[i, 0].set_title("Grayscale")

        ax[i, 1].imshow(pred_np)
        ax[i, 1].set_title("Generated")

        ax[i, 2].imshow(real_np)
        ax[i, 2].set_title("Ground Truth")

        for j in range(3):
            ax[i, j].axis("off")

    save_path = f"./Results/{model_name}.jpg"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"\nSaved comparison image to: {save_path}\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ECE 60131 Project - Image Colorization")
    parser.add_argument("--seed", type=int, default=8000, help="Random seed")
    parser.add_argument("--model", type=str, default="unet", choices=["resnet", "unet", "gan", "unetgan", "basicvae", "unetvae"])
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--weights", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.001)

    args = parser.parse_args()

    create_dirs()

    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == "train":
        path = "./Dataset/"
        paths = np.array(glob.glob(path + "/*.jpg"))
        rand_indices = np.random.permutation(len(paths))

        train_idx, val_idx = rand_indices[:3600], rand_indices[3600:]
        train_paths = paths[train_idx]
        val_paths = paths[val_idx]
        
        wandb.init(
                project="Image Colorization",  # set your W&B project name
                config={
                    "seed": args.seed,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "model": args.model,
                    "latent_dim": args.latent_dim                    
                    },
                tags=[f"{args.model}"],
                name=f"{args.model}_{args.batch_size}_{args.epochs}_{args.latent_dim}_{args.learning_rate}"
            )

        lpips = LPIPSWrapper(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        trainer_class = TRAINER_MAP[args.model]
        
        trainer = trainer_class(train_paths, val_paths, args.latent_dim, lpips,
                                epochs=100,
                                batch_size=args.batch_size,
                                learning_rate=args.learning_rate,
                                num_workers=2)
        trainer.train()
        wandb.finish()

    elif args.mode == "test":
        test(args.model, args.weights)