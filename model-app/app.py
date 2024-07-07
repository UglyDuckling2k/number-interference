import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# Check model file path
if os.getenv('RUNNING_IN_DOCKER') == 'true':
    model_path = '/model/digit_classifier.pth'
else:
    model_path = './model/digit_classifier.pth'
print(f"Model path: {model_path}")

writer = SummaryWriter('/logs')

# Define your CNN model
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)  # Adding dropout
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 32 * 14 * 14)
        x = self.dropout(x)  # Applying dropout
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 10

# Load MNIST dataset and prepare data loaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = MNIST(root='/data', train=True, download=True, transform=transform)
test_dataset = MNIST(root='/data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = DigitClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):

        # Ensure the labels and images are on the same device as the model
        images, labels = images.to(device), labels.to(device)  

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}")
            writer.add_scalar('training loss', running_loss / 100, epoch * len(train_loader) + i)
            running_loss = 0.0

# After training
writer.flush()
writer.close()

# Save the trained model
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
