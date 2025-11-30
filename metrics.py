import torch
import torch.nn.functional as F
from torchvision import transforms
from pytorch_msssim import ssim
from lpips import LPIPS
import numpy as np
from skimage import color


def compute_ssim(pred, target):                                         # structural similarity index
    return ssim(pred, target, data_range=1.0, size_average=True)


def compute_deltaE(pred, target):                                       # change in energy
    pred_np = pred.detach().cpu().permute(0, 2, 3, 1).numpy()
    tgt_np  = target.detach().cpu().permute(0, 2, 3, 1).numpy()

    total_de = 0
    for p, t in zip(pred_np, tgt_np):
        p_lab = color.rgb2lab(p)
        t_lab = color.rgb2lab(t)
        delta = color.deltaE_ciede2000(p_lab, t_lab)
        total_de += delta.mean()

    return total_de / pred_np.shape[0]


class LPIPSWrapper:                                                     # LPIPS
    def __init__(self, net='alex', device='cuda'):
        self.model = LPIPS(net=net).to(device)
        self.device = device

    def __call__(self, pred, target):
        pred_lp = (pred * 2 - 1).to(self.device)
        tgt_lp  = (target * 2 - 1).to(self.device)
        lp = self.model(pred_lp, tgt_lp)
        return lp.mean()