# Author: Imraj Singh

# First version: 21st of May 2022

# CCP SyneRBI Synergistic Image Reconstruction Framework (SIRF).
# Copyright 2022 University College London.

# This is software developed for the Collaborative Computational Project in Synergistic Reconstruction for Biomedical Imaging (http://www.ccpsynerbi.ac.uk/).

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


import torch
import numpy as np
from misc import random_phantom, shepp_logan

#%matplotlib inline
import argparse
import os
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

class EllipsesDataset(torch.utils.data.Dataset):

    """ Pytorch Dataset for simulated ellipses
    Initialisation
    ----------
    fwd_op : `SIRF acquisition model`
        The forward operator
    image template : `SIRF image data`
        needed to project and to get shape
    n_samples : `int`
        Number of samples    
    mode : `string`
        Type of data: training, validation and testing
    seed : `int`
        The seed used for the random ellipses
    """

    def __init__(self, image_template, n_samples = 100, mode="train", seed = 1, transform=None):
        #self.fwd_op = fwd_op
        
        self.image_template = image_template
        self.n_samples = n_samples
        self.tranfrom = transform

        if mode == 'valid':
            self.x_gt = shepp_logan(self.image_template.shape)
            #self.y = self.__get_measured__(self.x_gt)

        #self.primal_op_layer = fwd_op
        self.mode = mode
        np.random.seed(seed)

    # def __get_measured__(self, x_gt):
    #     # Forward project image then add noise
    #     y = self.fwd_op(self.image_template.fill(x_gt))
    #     y = np.random.poisson(y.as_array()[0])
    #     return y

    def __len__(self):
        # Denotes the total number of iters
        return self.n_samples

    def __getitem__(self, index):
        # Generates one sample of data
        if self.mode == "train":
            x_gt = random_phantom(self.image_template.shape)
            x_gt = x_gt.astype(np.float32)
            
            #y = self.__get_measured__(x_gt)

        elif self.mode == "valid":
            x_gt = self.x_gt
            x_gt = x_gt.astype(np.float32)
            #y = self.y

        else:
            NotImplementedError

        return x_gt
    
def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))
    

if __name__ == "__main__":

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    a = np.ones((64,64))
    ellipse = EllipsesDataset( image_template=a, n_samples=100, mode="train", seed=1)
    dataloader = torch.utils.data.DataLoader(ellipse, batch_size=4)

    batch = next(iter(dataloader))

    #batch = normalize(batch)
    print(batch.shape)

    # make a grid from batch
    grid = vutils.make_grid(batch)

    #print(batch.shape)

    # show images
    #plt.imsave('test.png', grid.numpy().transpose((1, 2, 0)))
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.show()





