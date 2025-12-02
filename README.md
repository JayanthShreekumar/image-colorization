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

Each Model folder has 2 files: 
    model.py which contains the model definition
    train.py which contains the training loop

