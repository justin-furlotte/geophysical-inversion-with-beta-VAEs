from __future__ import print_function

import torch
import torch.utils.data
from torch.utils.data import TensorDataset,DataLoader
from torch import nn, optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
import time
import numpy as np

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten65(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 256, 2, 2)


class VAE65(nn.Module):
    def __init__(self, gpath, image_channels=1, h_dim=12 * 256, z_dim=10,cuda=True):
        super(VAE65, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=5, stride=2, bias=False),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(4 * 256, 512),
            nn.LeakyReLU()
        )

        self.fc1 = nn.Linear(512, z_dim)
        self.fc2 = nn.Linear(512, z_dim)
        self.fc3 = nn.Linear(z_dim, 512)

        self.decoder = nn.Sequential(
            nn.Linear(512, 4 * 256),
            nn.LeakyReLU(),
            UnFlatten65(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, bias=False),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=5, stride=2, bias=False),
            nn.ReLU()
        )

        if cuda and gpath !=0:
            self.load_state_dict(torch.load(gpath))

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size(), device=device)
        z = mu + std * esp * alpha
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon_x = self.decode(z)
        return recon_x, z, mu, logvar

seed = 1
#Get GPU running
torch.manual_seed(seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
j=torch.cuda.is_available()

#Load synthetic Geology dataset
iocg_synth = torch.load("simple_layers_test.pt")
dataset = TensorDataset(iocg_synth)
dataloader = DataLoader(dataset,100)


#gpath = 'VAE65_10_a1b10_le_JW.pth'
gpath = 0 #Set to zero is starting training from scratch, otherwise set to file name.

#One input channel
image_channels = 1

#Initialize Model
model = VAE65(gpath=gpath,image_channels=image_channels).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


#model.load_state_dict(torch.load(gpath)) #Uncomment to Load in saved model if you want


def loss_fn(recon_x, x, mu, logvar,zencoded):
    #Mean square error
    MSE = F.mse_loss(recon_x, x, reduction='sum')

    #KLDivergence Term
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + beta*KLD, MSE, KLD

#Dimensionality of Latent Space
dimz = 10

#Parameters to weigh variance of noise, KL divergence
alpha = 1
beta = 100.0

epochs = 50
batch_size = 100

#Set model to train mode(As opposed to evaluate)
model.train()

t0 = time.time()
lhist = []
mhist = []
khist = []


for epoch in range(epochs):
    for i, data in enumerate(dataloader,0):
        if i >= 1000:
            break

        images_device = data[0].to(device)

        #Optimize Model
        recon_images, z_encoded, mu, logvar = model(images_device)
        loss, mse, kld = loss_fn(recon_images, images_device, mu, logvar,z_encoded)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # if idx%100 == 0:
    to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch + 1,
                                                                epochs, loss.item() / batch_size,
                                                                mse.item() / batch_size, kld.item() / batch_size)
    print(to_print)
    lhist.append(loss.item() / batch_size)
    mhist.append(mse.item() / batch_size)
    khist.append(kld.item() / batch_size)

tf = time.time() - t0
stf = time.strftime("%H:%M:%S", time.gmtime(tf))
print('\ncomputation time: {0}'.format(stf))

#Save as parameters, 10 latent components, alpha=1, beta=10, layered earth
torch.save(model.state_dict(), 'VAE65_10_a1b10_le_JW.pth')

#Plot reconstruction
my_x = data[0]
plt.figure()
indt = 14
vmin,vmax = 0.0,1.0
plt.imshow(my_x[indt,0,:,:], vmin=vmin,vmax=vmax)
plt.show()
imgtens = torch.Tensor(my_x)
recon_x,zen,mn,vari = model(imgtens.to(device))
xx = recon_x.cpu().detach().numpy()
plt.figure()
plt.imshow(xx[indt,0,:,:],  vmin=vmin,vmax=vmax)
plt.show()

#Plot randomly generated samples
z_new = 1.0*torch.randn(5,dimz, dtype=torch.float, device=device)
x_new = model.decode(z_new)
x_new = x_new.cpu().detach().numpy()

fig, ax = plt.subplots(1,5,figsize=(15,3))
ax = ax.ravel()
vmax,vmin = 1.0,0.0
vmax,vmin = np.max(x_new),np.min(x_new)
for i in range(len(ax)):
    ax[i].imshow(x_new[i,0,:,:],vmin=vmin,vmax=vmax)
plt.show()

