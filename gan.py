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
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28), num_classes=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# %%
img_shape = (1, 32, 32)
latent_dim = 100

generator = Generator(latent_dim=latent_dim, img_shape=img_shape)
discriminator = Discriminator(img_shape=img_shape)

#%%
adversarial_loss = nn.CrossEntropyLoss()

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
    d_loss_total = 0
    g_loss_total = 0
    for i, (imgs, labels) in enumerate(train_loader):
        batch_size = imgs.size(0)
        
        # Create labels
        real_labels = labels  # 0 or 1 for real diseases
        fake_labels = torch.full((batch_size,), 2, dtype=torch.long)  # 2 for fake images

        real_imgs = imgs

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        z = torch.randn(batch_size, latent_dim)
        gen_imgs = generator(z)

        fake_output = discriminator(gen_imgs)
        g_loss = adversarial_loss(fake_output, real_labels)  # Try to fool discriminator

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        real_output = discriminator(real_imgs)
        fake_output = discriminator(gen_imgs.detach())

        real_loss = adversarial_loss(real_output, real_labels)
        fake_loss = adversarial_loss(fake_output, fake_labels)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        d_loss_total += d_loss.item()
        g_loss_total += g_loss.item()
    print(f"Epoch {epoch+1}/{n_epochs} - D Loss: {d_loss_total/len(train_loader)} - G Loss: {g_loss_total/len(train_loader)}")
        

#%% predict on test set

#%% predict on test set
correct_predictions = 0
total_samples = 0

with torch.no_grad():
    for i, (imgs, labels) in enumerate(test_loader):
        outputs = discriminator(imgs)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        print(f"True label: {labels.item()}, Predicted: {predicted.item()}")

accuracy = 100 * correct_predictions / total_samples
print(f"Test Accuracy: {accuracy:.2f}%")

# %%
