"""
*    Title: PyTorch DCGAN Tutorial
*    Author: Nathan Inkawhich
*    Date: 2023
*    Availability: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
*    GitHub Availability: https://github.com/pytorch/examples/blob/main/dcgan/main.py
"""


from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm


from ellipses import EllipsesDataset
from network.Generator import Generator
from network.Discriminator import Discriminator

torch.cuda.empty_cache()

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 128

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 100

# Learning rate for optimizers
lr = 0.0008

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# number of iteratiion
niter = batch_size * 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# tranformation for the data loader
transform = transforms.Compose([
                                transforms.Normalize((0.5,), (0.5,)),
])


image_template = np.zeros((image_size, image_size))
ellipse_dataset = EllipsesDataset(image_template, n_samples = niter, mode="train", seed = 1)
train_loader = torch.utils.data.DataLoader(ellipse_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    
# Create the generator
netG = Generator(nz, ngf, nc).to(device)

# Apply the weights_init function to randomly initialize all weights
netG.load_state_dict(torch.load('trained_models/gen.pth'))

# Print the model
print(netG)
    
# Create the Discriminator
netD = Discriminator(nc, ndf).to(device)

# Apply the weights_init function to randomly initialize all weights
netD.load_state_dict(torch.load('trained_models/dis.pth'))


# Print the model
print(netD)

# Initialize BCELoss function
criterion = nn.BCELoss()
#criterion = nn.MSELoss()    

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
i = 0
for epoch in range(num_epochs):
    # For each batch in the dataloader
    #for i in tqdm(range(niter)):
    
    for data in tqdm(train_loader):

        #data = torch.from_numpy(tomo_data.create_batch(batch_size))
        gaussian_blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        data = gaussian_blur(data)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data.to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D

        
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D

        

        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, i,
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
        i += 1

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

    # save an image of the generator output with the fixed noise
    if epoch % 10 == 0:
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            # save image to gen_img folder
            vutils.save_image(fake, 'data/gen_img/epoch_%d.png' % epoch)


# save the model
torch.save(netG.state_dict(), 'data/save_models/gen.pth')
torch.save(netD.state_dict(), 'data/save_models/dis.pth')




plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('data/losses.png')
plt.show()

