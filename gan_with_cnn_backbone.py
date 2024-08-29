#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np

# %%
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 32, 32)):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        
        self.init_size = img_shape[1] // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, img_shape[0], 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape=(1, 32, 32)):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, bn=True, kernel_size=4, stride=2, padding=1):
            block = [nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            return block
        
        self.model = nn.Sequential(
            *discriminator_block(img_shape[0], 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        
        # The height and width of downsampled image
        ds_size = img_shape[1] // 2**3
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

# %%
img_shape = (1, 32, 32)
latent_dim = 100

generator = Generator(latent_dim=latent_dim, img_shape=img_shape)
discriminator = Discriminator(img_shape=img_shape)

#%%
adversarial_loss = torch.nn.BCELoss()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

#%%
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

trainset = torchvision.datasets.ImageFolder(root='data/train', transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.ImageFolder(root='data/test', transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)



#%%
n_epochs = 10
for epoch in range(n_epochs):
    d_loss_total = []
    g_loss_total = []
    for i, (imgs, _) in enumerate(train_loader):
        valid = torch.ones(imgs.size(0), 1, requires_grad=False)
        fake = torch.zeros(imgs.size(0), 1, requires_grad=False)

        real_imgs = imgs

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        z = torch.randn(imgs.shape[0], latent_dim)
        gen_imgs = generator(z)

        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss_total.append(g_loss.item())
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss_total.append(d_loss.item())
        d_loss.backward()
        optimizer_D.step()

    print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(train_loader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

#%% predict on test set
z = torch.randn(1, latent_dim)
gen_imgs = generator(z)

# %% visualise generated image
import matplotlib.pyplot as plt
plt.imshow(gen_imgs.squeeze().detach().numpy(), cmap='gray')
plt.show()
# %% visualise real image
real_imgs, _ = next(iter(test_loader))
plt.imshow(real_imgs.squeeze().detach().numpy(), cmap='gray')


# %% check for test set accuracy


