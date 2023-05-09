import torch 
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

from ellipses import EllipsesDataset
from network.Generator import Generator
from tqdm import tqdm
from skimage.transform import radon, iradon, iradon_sart

import sys

# set the terminal display to inf 


class ForwardPhysicalModel(torch.nn.Module):
  def __init__(self, views_theta):
    super(ForwardPhysicalModel, self).__init__()
    self.views_theta = views_theta

  def forward(self, x):
    return radon_forward.apply(x, self.views_theta)

class radon_forward(torch.autograd.Function):
  @staticmethod
  def forward(ctx, image, views_theta):
    ctx.views_theta = views_theta
    out = torch.from_numpy(radon(image.detach().cpu().numpy(), theta=views_theta))
    out.requires_grad = True
    return out.to(device)

  @staticmethod
  def backward(ctx, data):
    out = torch.from_numpy(iradon(data.detach().cpu().numpy(), \
                                  theta=ctx.views_theta, filter_name=None, \
                                  interpolation='linear', circle=True, \
                                  preserve_range=True))
    out.requires_grad = True
    return out.to(device), None

def PSNR(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = torch.max(img1)
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


# Load the generator
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
path = "trained_models/gen.pth"
netG = Generator(128, 64, 1)#.to(device)
netG.load_state_dict(torch.load(path))


# Generate a ground truth  ellipse
image_template = np.ones((64, 64))
ellipse_dataset = EllipsesDataset(image_template = image_template, n_samples = 5, mode="train", seed = 9) # 1 , 2  5 7
ellipse = torch.from_numpy(ellipse_dataset[0][0])#.to(device)

# return the biggest value in ellipse tensor
max_value = torch.max(ellipse)

# Normalise the ellipse tensor
ellipse = ellipse/max_value





views_theta = np.linspace(0., 180., 30, endpoint=False)
gt_sinogram = radon(ellipse.cpu(), theta=views_theta)
print(type(gt_sinogram))
noisy_gt_sinogram = gt_sinogram + np.random.rand(*gt_sinogram.shape)*2
noisy_gt_sinogram = noisy_gt_sinogram.astype(np.float32)

# apply fbp to the noisy sinogram
fbp_noisy_gt = iradon(noisy_gt_sinogram, theta=views_theta, circle=True)

# apply sart to the noisy sinogram
sart_noisy_gt = iradon_sart(noisy_gt_sinogram, theta=views_theta)

fbp_mse = torch.nn.functional.mse_loss(torch.from_numpy(fbp_noisy_gt), ellipse.cpu())
sart_mse = torch.nn.functional.mse_loss(torch.from_numpy(sart_noisy_gt), ellipse.cpu())

fbp_psnr = PSNR(torch.from_numpy(fbp_noisy_gt), ellipse.cpu())
sart_psnr = PSNR(torch.from_numpy(sart_noisy_gt), ellipse.cpu())

print("FBP MSE: ", fbp_mse)
print("SART MSE: ", sart_mse)

print("FBP PSNR: ", fbp_psnr)
print("SART PSNR: ", sart_psnr)

breakpoint()

z = torch.nn.parameter.Parameter(torch.randn(70, 128, 1, 1))
optimizer = torch.optim.Adam([z], lr=0.1)

fwdmodel = ForwardPhysicalModel(views_theta).to(device)


noisy_gt_sinogram_torch = torch.from_numpy(noisy_gt_sinogram).to(device)

weight = 0


psnr_list = []
mse_list = []
for i in range(35):
    for i in tqdm(range(100)):
        
        optimizer.zero_grad()
        reconstruction = netG(z).to(device)
        # map recinstructed image to [0, 1]
        
        estimate_sinogram = fwdmodel(reconstruction[0].squeeze(0)).to(device)

        loss = torch.nn.functional.mse_loss(estimate_sinogram, noisy_gt_sinogram_torch) + weight * i * (np.linalg.norm(z[0].detach().numpy()))**2 # add the l2 noram

        loss.backward() 
        optimizer.step()
    
    weight += 1
    psnr_list.append(PSNR(reconstruction[0][0].cpu().detach(), ellipse.cpu().detach()))
    mse_list.append(torch.nn.functional.mse_loss(reconstruction[0][0].cpu().detach(), ellipse.cpu().detach()))

print("PSNR: ", psnr_list)
print("MSE: ", mse_list)

# save psnr and mse list as a text file
np.savetxt("psnr_list.txt", psnr_list)
np.savetxt("mse_list.txt", mse_list)

# visualise the psnr
fig = plt.figure(figsize=(10, 10))
plt.title("PSNR against Weight")
plt.xlabel("Weight")
plt.ylabel("PSNR")
plt.plot(psnr_list)
plt.show()


fig = plt.figure(figsize=(10, 10))
plt.title("MSE against Weight")
plt.xlabel("Weight")
plt.ylabel("MSE")
plt.plot(mse_list)
plt.show()


# fbp_noisy_gt = iradon(noisy_gt_sinogram, theta=views_theta, circle=True)
# max_value_fbp = np.max(fbp_noisy_gt)
# fbp_noisy_gt = fbp_noisy_gt/max_value_fbp

# max_reconstruction = torch.max(reconstruction[0][0])
# reconstruction[0][0] = reconstruction[0][0]/max_reconstruction

# fig = plt.figure(figsize=(10, 10))
# fig.add_subplot(1, 3, 1)
# plt.imshow(reconstruction[0][0].cpu().detach().numpy(), cmap='gray')
# plt.colorbar(label="colour bar", orientation="horizontal")
# plt.title("Reconstruction")


# fig.add_subplot(1, 3, 2)
# plt.imshow(fbp_noisy_gt, cmap='gray')
# plt.colorbar(label="colour bar", orientation="horizontal")
# plt.title("FBP")

# fig.add_subplot(1, 3, 3)
# plt.imshow(ellipse, cmap='gray')
# plt.colorbar(label="colour bar", orientation="horizontal")
# plt.title("Ground Truth")

# plt.show()