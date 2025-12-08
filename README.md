
The in-class presentation slides are uploaded here as a PDF file - ```Image Colorization.pdf```.

**CODEBASE**

```bash
# Image Colorization using Deep Learning
# Folder Structure
./image-colorization/
├── BasicVAE
├── Dataset                 # Contains all images for training
├── GAN
├── Models                  #Store model weights after training
│   ├── basicvae
│   ├── gan
│   ├── resnet
│   ├── unet
│   ├── unetddim
│   ├── unetddpm
│   ├── unetgan
│   └── unetvae
├── ResNet
├── Results                 # Store output images after inference
├── TestImages              # Contains all images for inference
├── UNet
├── UNetDiffusion           # Contains both DDPM and DDIM code for training, model is the same
├── UNet_GAN
├── UNetVAE
└── wandb
.gitignore
colorize_data.py            # Dataloader, takes in color images and creates grayscale version for training
environment.yml             # .yml file for installing all required packages
main.py                     # main script to run
metrics.py                  # Metrics code for calculation
nohup.out                   # Stores output of runs when running in background
README.md                   # Instructions
script.sh                   # Script to run all models
```

Each Model folder has 2 files: \
    model.py which contains the model definition \
    train.py which contains the training loop 

The main.py file takes in arguments from script.sh and creates and trains the chosen model accordingly.

The runs are recorded for comparison using ```wandb```. An account is needed to monitor the dashboard online: https://docs.wandb.ai/models/quickstart#python.

```script.sh``` contains bash code to start training and inference of all codes. Since training takes a while, all runs are pushed to the backend.

These are all the arguments required or optional for the argparser:

```python
parser.add_argument("--seed", type=int, default=8000, help="Random seed")
parser.add_argument("--model", type=str, default="unet", choices=["resnet", "unet", "gan", "unetgan", "basicvae", "unetvae", "unetddpm", "unetddim"])
parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
parser.add_argument("--weights", type=int, default=100)                 # which saved model weights to use for inference
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--latent_dim", type=int, default=256)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--steps", type=int, default=50)                    # for ddim
parser.add_argument("--num_images", type=int, default=4)                # to plot for inference
```
---

**ENVIRONMENT CREATION**

The project was implemented in python 3.12 using Anaconda. Create an environment using the ```environment.yml``` file.

----------------------------------------------------------------------------------------

**DATASET**

The dataset is a small one with 3600 training images, 618 validation images, and 64 testing images. Download the dataset using the link below:\
https://drive.google.com/file/d/1vSsSQrMooCxrJHV5PhsJhAz1H_QJOShQ/view?usp=drive_link

Train images are stored in the ```Dataset``` folder
Separate out a couple test images for inference (around 100, in no particular order) and store them in ```TestImages```.

------------------------------------------------------------------------------------------

**PROJECT IDEA**

Image colorization is the process of converting a grayscale image into a realistic color image by predicting plausible colors for every pixel.

It is a one-to-many problem: multiple colorizations can be valid for the same grayscale input – clothes a person is wearing, color of leaves in different seasons, etc.​.

A successful model needs semantic understanding — it must recognize objects like sky, skin, foliage, buildings, clothing, etc. and constrain the color space.

Classical models fail because they rely on low-level cues instead of semantic understanding - deep learning models, especially the state of the art, understand image semantics and are capable of resolving semantic ambiguity.

The project lends itself to many generative models by its nature - generating color from scratch. 

-------------------------------------------------------------------------------------------------

**METHODOLOGY**

1) ResNet, UNet were trained using MSE loss. Since ResNet only performs the Conv2D, a sequential upsampling phase using Conv2DTranspose was used to generate color images.  UNet naturally lends itself to the project since it incorporates a downsampling as well as an upsampling phase in its architecture. UNet produced significantly better results due to the skip connections which helped keep structure intact.

2) GAN, UNetGAN were trained using binary cross entropy loss (both for the discriminator and the generator). The GAN generator consisted of a downsampling phase and an upsampling phase to generate color images without skip connections. The UNetGAN generator uses a UNet architecture. The discriminator for both architectures consisted on Conv2D layers for downsampling.

$$
\text{BCE}(y, \hat{y}) = - \big( y \cdot \log \hat{y} + (1 - y) \cdot \log (1 - \hat{y}) \big)
$$

3) BasicVAE and UNetVAE were trained using the standard VAE loss that consists of a reconstruction term and a regularization term:

$$
\mathcal{L} = \underbrace{\text{MSE}(x, \hat{x})}_{\text{reconstruction}} + 0.001 \cdot \underbrace{D_{\mathrm{KL}}\!\left(q(z|x)\,\|\,p(z)\right)}_{\text{regularization}}
$$

$$\mathcal{L}
= \frac{1}{N} \sum_{i=1}^{N} \lVert \hat{x}_i - x_i \rVert_2^{2}
\;+\;
0.001 \left(
-\frac{1}{2} \cdot \frac{1}{N} \sum_{i=1}^{N}
\left( 1 + \log\sigma_i^{2} - \mu_i^{2} - \sigma_i^{2} \right)
\right)
$$

    a. Reconstruction Term Intuition​
   
            This tells the decoder: “Reconstruct the input as accurately as possible.”​  
            It forces the latent code to contain useful information about the input.​   
            Without it, the model would ignore the input and generate meaningless samples.​   
    ​
    b. KL Divergence Term Intuition​
   
            Encourages the learned latent distribution to stay close to a simple prior p(z)=N(0,I).​   
            Prevents the model from memorizing or overfitting the dataset.​

4) Implemented Diffusion models (both DDPM and DDIM) but without much success. The training of DDPM took 2 days and it did not provide any visualizable results while DDIM took about a day with the same problem.
       
------------------------------------------------------------------------------------------

**CONTRIBUTION**

The main goal of the project was to gain experience building and training several generative models from scratch and compare them.

This codebase contains useful model and training code for a plethora of deep generative models that can be modified for your task.

------------------------------------------------------------------------------------------

**METRICS USED**

Structural similarity index measure (SSIM), Learned Perceptual Image Patch Similarity (LPIPS), ​Delta E in LAB space(ΔE)​.

------------------------------------------------------------------------------------------------------------------------

**RESULTS**

| **Model**   | **Loss (no meaning)** | **SSIM (↑)** | **LPIPS (↓)** | **ΔE (↓)**   |
|-------------|-----------------------|--------------|---------------|--------------|
| **ResNet**  | 0.213                 | _0.492_      | 0.428         | 16.533       |
| **UNet**    | 0.181                 | **_0.678_**  | **_0.242_**   | **_15.350_** |
| **GAN**     | 13.686                | 0.476        | 0.463         | _15.451_     |
| **UNetGAN** | 108.944               | **0.728**    | **0.239**     | **14.386**   |
| **VAE**     | 0.212                 | 0.175        | 0.6732        | 21.834       |
| **UNetVAE** | 0.197                 | 0.312        | _0.427_       | 19.509       |

Best results are in bold, 2nd best results are in italics, and 3rd best results are in bold and italics.

Sample generated color images are in the ```Results``` folder.

-------------------------

**FUTURE WORK**

Train diffusion models and evaluate them thoroughly.

----------------------------
