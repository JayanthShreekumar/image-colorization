```bash
# Image Colorization using Deep Learning
**Folder Structure**
./image-colorization/
├── BasicVAE
├── Dataset                 <!-- Contains all images for training -->
├── GAN
├── Models                  <!-- Store model weights after training -->
│   ├── basicvae
│   ├── gan
│   ├── resnet
│   ├── unet
│   ├── unetddim
│   ├── unetddpm
│   ├── unetgan
│   └── unetvae
├── ResNet
├── Results                 <!-- Store output images after inference -->
├── TestImages              <!-- Contains all images for inference -->
├── UNet
├── UNetDiffusion           <!-- Contains both DDPM and DDIM code for training, model is the same -->
├── UNet_GAN
├── UNetVAE
└── wandb
.gitignore
colorize_data.py            <!-- Dataloader, takes in color images and creates grayscale version for training -->
environment.yml             <!-- .yml file for installing all required packages -->
main.py                     <!-- main script to run -->
metrics.py                  <!-- Metrics code for calculation -->
nohup.out                   <!-- Stores output of runs when running in background -->
README.md                   <!-- Instructions -->
script.sh                   <!-- Script to run all models -->
```

The dataset is a small one with 3600 training images, 618 validation images, and 64 testing images.

Each Model folder has 2 files: \
    model.py which contains the model definition \
    train.py which contains the training loop 

The main.py file takes in arguments from script.sh and creates and trains the chosen model accordingly.\
environment.yml contains the packages needed to run the experiments using Anaconda. All experiments were tested in python 3.12.

The runs are recorded for comparison using wandb. An account is needed to monitor the dashboard online: https://docs.wandb.ai/models/quickstart#python.

The in-class presentation slides are uploaded here as a PDF file.

METRICS USED: Structural similarity index measure (SSIM), Learned Perceptual Image Patch Similarity (LPIPS), ​Delta E in Lab space (ΔE)​.

TODO: Train diffusion models and evaluate them thoroughly.

