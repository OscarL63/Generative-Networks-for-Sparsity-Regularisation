import torch 
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

from ellipses import EllipsesDataset
from network.Generator import Generator
from tqdm import tqdm
from skimage.transform import radon, iradon


img_list = []
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
path = "trained_models/gen.pth"
netG = Generator(128, 64, 1).to(device)
netG.load_state_dict(torch.load(path))

noise = torch.randn(1, 128, 1, 1, device = device)
fake = netG(noise)
print(fake.shape)

image_template = np.ones((64, 64))

ellipse_dataset = EllipsesDataset(image_template = image_template, n_samples = 1, mode="train", seed = 7) #56 1,2,5,7


z = torch.nn.parameter.Parameter(torch.randn(70, 128, 1, 1, device = device))
optimizer = torch.optim.Adam([z], lr=0.1)

ellipse = torch.from_numpy(ellipse_dataset[0][0]).unsqueeze(0).to(device)

# return the biggest value in ellipse tensor
max_value = torch.max(ellipse)

# Normalise the ellipse tensor
ellipse = ellipse/max_value

gt_ellipse = ellipse

gaussian_blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
ellipse = gaussian_blur(ellipse)

ellipse_numpy = ellipse.cpu().detach().squeeze(0).numpy()

views_theta = np.linspace(0., 180., 30, endpoint=False)
measured = radon(ellipse_numpy, theta=views_theta, circle=True)
fbp = iradon(measured, theta=views_theta, circle=True)
max_value_fbp = np.max(fbp)
fbp = fbp/max_value_fbp

for i in tqdm(range(1000)):
    
    optimizer.zero_grad()
    reconstruction = netG(z)
    loss = torch.nn.functional.mse_loss(reconstruction[0], ellipse)
    loss.backward()
    optimizer.step()


max_reconstruction = torch.max(reconstruction[0][0])
reconstruction[0][0] = reconstruction[0][0]/max_reconstruction



fig = plt.figure(figsize=(10, 10))
fig.add_subplot(1, 3, 1)
plt.imshow(reconstruction[0][0].cpu().detach().numpy(), cmap='gray')
plt.colorbar(label="colour bar", orientation="horizontal")
plt.title("GAN inversion")

fig.add_subplot(1, 3, 3)
plt.imshow(gt_ellipse[0].cpu().detach().numpy(), cmap='gray')
plt.colorbar(label="colour bar", orientation="horizontal")
plt.title("Ground truth")

fig.add_subplot(1, 3, 2)
plt.imshow(fbp, cmap='gray')
plt.colorbar(label="colour bar", orientation="horizontal")
plt.title("FBP")

plt.show()




    



