'''
conda deactivate
conda deactivate
CUDA_VISIBLE_DEVICES=1
cd autoencoder_attacks/
cd train_aautoencoders/
conda activate inn
python vae_celebA_three_non_reduction_layers_THIS_ONE_New.py
'''

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn, optim
from torchvision import datasets, transforms

import zipfile

import shutil
import os
import pandas as pd

device = ("cuda:1" if torch.cuda.is_available() else "cpu") # Use GPU or CPU for training



root = 'data_faces/img_align_celeba'
img_list = os.listdir(root)
print(len(img_list))
     


df = pd.read_csv("list_attr_celeba.csv")
df = df[['image_id', 'Smiling']]



img_list = os.listdir('data_cel1/smile/')
img_list.extend(os.listdir('data_cel1/no_smile/'))


transform = transforms.Compose([
          transforms.Resize((64, 64)),
          transforms.ToTensor()
          ])

batch_size = 64
celeba_data = datasets.ImageFolder('data_cel1', transform=transform)



train_set, test_set = torch.utils.data.random_split(celeba_data, [int(len(img_list) * 0.8), len(img_list) - int(len(img_list) * 0.8)])
train_data_size = len(train_set)
test_data_size = len(test_set)


trainLoader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True)
testLoader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)


import torch
import torch.nn as nn
import torch.nn.functional as F

from vae import VAE, VAE_big



image_channels = 3

print('image_channels', image_channels)

#model = VAE(image_channels=image_channels).to(device)

model = VAE_big(device, image_channels=image_channels).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 

beta_value = 9.0

def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + beta_value *  KLD, BCE, KLD



epochs = 200



train_loss = []

for epoch in range(epochs):
   
    total_train_loss = 0
    # training our model
    for idx, (image, label) in enumerate(trainLoader):
        images, label = image.to(device), label.to(device)

        recon_images, mu, logvar = model(images.to(device))
        loss, bce, kld = loss_fn(recon_images.to(device), images.to(device), mu.to(device), logvar.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    print('loss', loss)
    print("Epoch : ", epoch)


    torch.save(model.state_dict(), '/home/luser/autoencoder_attacks/saved_celebA/checkpoints/celebA_CNN_VAE'+str(beta_value)+'_big_trainSize'+str(train_data_size)+'_epochs'+str(epoch)+'.torch')