import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from compactlbfgs import LayerWiseCompactLBFGS

# CIFAR-10 Data Loaders
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_loader = DataLoader(
    torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform),
    batch_size=128, shuffle=True, num_workers=0)

test_loader = DataLoader(
    torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform),
    batch_size=128, shuffle=False, num_workers=0)

# Model (simulate ResNet-100 using ResNet-50 for now)
class ResNetSimulated(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50()
        self.model.fc = nn.Linear(2048, 10)

    def forward(self, x):
        return self.model(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetSimulated().to(device)

# Optimizer
optimizer = LayerWiseCompactLBFGS(model.parameters(), lr=1e-3, history_size=10)
criterion = nn.CrossEntropyLoss()

model.train()
# Training loop
for epoch in range(5):  # Short training for demo
    
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        def closure():
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Training Loss: {total_loss:.4f}")
