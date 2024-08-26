#%% Semi-Supervised Learning
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import random
from sklearn.metrics import accuracy_score
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

# %% Hyperparameters
BATCH_SIZE = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 10
LOSS_FACTOR_SELFSUPERVISED = 1  # Reduced from 1 to 0.1
LEARNING_RATE = 0.01  # Reduced from 0.001 to 0.0001

# %% image transformation steps
transform_super = transforms.Compose(
    [transforms.Resize(32),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])  # Corrected normalization for grayscale


class UnlabeledDataset(Dataset):
    """Create the Dataset for Unlabeled Data

    Args:
        Dataset (_type_): _description_
    """
    sesemi_transformations = {0: 0, 1: 90, 2: 180, 3: 270}
        
    def __init__(self, folder_path) -> None:
        super().__init__()
        self.folder_path = folder_path
        self.image_names = os.listdir(folder_path)
        self.images_full_path_names = [f"{folder_path}/{i}" for i in self.image_names]
        
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img = Image.open(self.images_full_path_names[idx])
        # get a random sesemi transformation
        transformation_class_label = random.randint(0, len(self.sesemi_transformations) - 1)
        # apply the randomly selected transformation
        angle = self.sesemi_transformations[transformation_class_label]
        
        data = transform_super(img)        
        data = transforms.functional.rotate(img=data, angle=angle)       
        return data, transformation_class_label

# %% Dataset for unlabeled data
folder_path = 'data/unlabeled'
unlabeled_ds = UnlabeledDataset(folder_path)

#%% Dataset for train and test
train_ds = torchvision.datasets.ImageFolder(root='data/train', transform=transform_super)
test_ds = torchvision.datasets.ImageFolder(root='data/test', transform=transform_super)
# %% Dataloaders
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_ds, batch_size=BATCH_SIZE, shuffle=True)

#%% Model Class
class SesemiNet(nn.Module):
    def __init__(self, n_super_classes, n_selfsuper_classes) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out_super = nn.Linear(64, n_super_classes)
        self.fc_out_selfsuper = nn.Linear(64, n_selfsuper_classes)
        self.relu = nn.ReLU()
        
    def backbone(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x
    
    def forward(self, x):
        x = self.backbone(x)
        x_supervised = self.fc_out_super(x)
        x_selfsupervised = self.fc_out_selfsuper(x)
        return x_supervised, x_selfsupervised

model = SesemiNet(n_super_classes=2, n_selfsuper_classes=4).to(DEVICE)
model.train()

# %% Loss functions and Optimizer
criterion_supervised = nn.CrossEntropyLoss()
criterion_selfsupervised = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# %% Training loop
train_losses_super = []
train_losses_self = []
for epoch in range(NUM_EPOCHS):
    model.train()  # Set model to training mode
    train_loss_super = 0
    train_loss_self = 0
    for i, ((X_super, y_super), (X_selfsuper, y_selfsuper)) in enumerate(zip(train_loader, unlabeled_loader)):
        X_super, y_super = X_super.to(DEVICE), y_super.to(DEVICE)
        X_selfsuper, y_selfsuper = X_selfsuper.to(DEVICE), y_selfsuper.to(DEVICE)
        
        optimizer.zero_grad()
        
        y_super_pred, _ = model(X_super)
        _, y_selfsuper_pred = model(X_selfsuper)
        
        loss_super = criterion_supervised(y_super_pred, y_super)
        loss_selfsuper = criterion_selfsupervised(y_selfsuper_pred, y_selfsuper)
        loss = loss_super + loss_selfsuper * LOSS_FACTOR_SELFSUPERVISED
        
        loss.backward()
        optimizer.step()
        
        train_loss_super += loss_super.item()
        train_loss_self += loss_selfsuper.item()
    
    train_losses_super.append(train_loss_super / len(train_loader))
    train_losses_self.append(train_loss_self / len(unlabeled_loader))
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Supervised Loss {train_losses_super[-1]:.4f}, Self-supervised Loss {train_losses_self[-1]:.4f}")

# %% Evaluation

y_test_preds = []
y_test_trues = []
with torch.no_grad():
    for X_test, y_test in test_loader:
        X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)
        y_test_pred, _ = model(X_test)
        
        y_test_pred_argmax = torch.argmax(y_test_pred, dim=1)
        y_test_preds.extend(y_test_pred_argmax.cpu().numpy())
        y_test_trues.extend(y_test.cpu().numpy())
        print(f"y_test: {y_test}, y_test_pred: {y_test_pred_argmax}")

accuracy = accuracy_score(y_pred=y_test_preds, y_true=y_test_trues)
print(f"Test Accuracy: {accuracy:.4f}")

# %% Plot training losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses_super, label='Supervised Loss')
plt.plot(train_losses_self, label='Self-supervised Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.show()
# %%