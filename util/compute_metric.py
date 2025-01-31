from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm
import matplotlib.pyplot as plt
import lpips
import numpy as np
import torch
from matric import calculate_ssim
import subprocess

device = 'cuda:0'
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

size=100

label_p=f'/label'
label_root = Path(label_p)
recon_p=f'/recon'
delta_recon_root = Path(recon_p)

psnr_delta_list = []
lpips_delta_list = []
ssim_delta_list=[]
for idx in tqdm(range(size)):
    fname = str(idx).zfill(5)

    label = plt.imread(label_root / f'{fname}.png')[:, :, :3]
    delta_recon = plt.imread(delta_recon_root / f'{fname}.png')[:, :, :3]
    psnr_delta = peak_signal_noise_ratio(label, delta_recon)
    psnr_delta_list.append(psnr_delta)
    ssim_delta_list.append(calculate_ssim(delta_recon,label))

    delta_recon = torch.from_numpy(delta_recon).permute(2, 0, 1).to(device)
    label = torch.from_numpy(label).permute(2, 0, 1).to(device)
    delta_recon = delta_recon.view(1, 3, 256, 256) * 2. - 1.
    label = label.view(1, 3, 256, 256) * 2. - 1.
    delta_d = loss_fn_vgg(delta_recon, label)
    lpips_delta_list.append(delta_d)

psnr_delta_avg = sum(psnr_delta_list) / len(psnr_delta_list)
lpips_delta_avg = sum(lpips_delta_list) / len(lpips_delta_list)
ssim_delta_avg=sum(ssim_delta_list) / len(ssim_delta_list)

print(label_p)
print(recon_p)
print(f'Delta PSNR: {psnr_delta_avg}')
print(f'Delta SSIM: {ssim_delta_avg}')
print(f'Delta LPIPS: {lpips_delta_avg}')
command = ['python', '-m', 'pytorch_fid', label_p, recon_p,'--batch-size',f'{size}']
result = subprocess.run(command, capture_output=True, text=True)
#FID
print(result.stdout) 
