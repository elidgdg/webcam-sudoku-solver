from torch.utils.data import DataLoader, ConcatDataset, Dataset # PyTorch data loader module
from torchvision import datasets # PyTorch vision module
from torchvision.transforms import ToTensor, transforms # Transform the data into PyTorch Tensors
import torch # PyTorch module
import torch.nn as nn # PyTorch neural network module
import torch.nn.functional as F # PyTorch functional module
import torch.optim as optim # PyTorch optimization module
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

        self.data = []
        self.labels = []

        # Load the data
        for i in range(10):
            for d in os.listdir(os.path.join(root_dir, str(i))):
                img_path = os.path.join(root_dir, str(i), d)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                self.data.append(self.transform(img))
                self.labels.append(i)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Load MNIST data
mnist_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

custom_data = CustomDataset("assets")

combined_data = ConcatDataset([mnist_data, custom_data])
combined_train, combined_test = train_test_split(combined_data, test_size=0.2, random_state=21)

loaders = {
    "train": DataLoader(combined_train, batch_size=64),
    "test": DataLoader(combined_test, batch_size=64)
}

class DigitRecogNetwork(nn.Module):
    def __init__(self, num_classes):
        super(DigitRecogNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 30, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(30, 15, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(15 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 50)
        self.fc3 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

# Instantiate the model
num_classes = 10 # 0-9
model = DigitRecogNetwork(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(10):
    for i, data in enumerate(loaders["train"]):
        # Get the inputs
        inputs, labels = data

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                  .format(epoch + 1, 10, i + 1, len(loaders["train"]), loss.item()))
            
# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data in loaders["test"]:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print("Accuracy of the model on the 10000 test images: {} %".format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), "model.pth")