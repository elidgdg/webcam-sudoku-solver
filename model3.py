import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms
from PIL import Image
import pandas as pd

# Load CSV files and concatenate them into a single DataFrame
csv_files = ["path/to/csv1.csv", "path/to/csv2.csv"]  # Specify the paths to your CSV files
df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

# Check the structure of your CSV file and adjust the column names accordingly
image_paths = df['image_path_column_name'].tolist()  # Adjust column name
labels = df['label_column_name'].tolist()  # Adjust column name

# Define the custom dataset class
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('L')  # 'L' for grayscale
        image = self.transform(image)
        label = self.labels[index]
        return image, label

# Create instances of the custom dataset
custom_dataset = CustomDataset(image_paths, labels, transform=ToTensor())

# Define the data loaders
loaders = {
    'train': DataLoader(custom_dataset, batch_size=100, shuffle=True, num_workers=1),
    'test': DataLoader(custom_dataset, batch_size=100, shuffle=True, num_workers=1)
}

# Define the neural network model class
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  # Use log_softmax for numerical stability

# Create the model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10  # Adjust based on your dataset
model = Net(num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Define the training function
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)} / {len(loaders["train"].dataset)} ({100. * batch_idx / len(loaders["train"]):.0f}%)]\tLoss: {loss.item():.6f}')

# Define the test function
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(loaders['test'].dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loaders["test"].dataset)} ({100. * correct / len(loaders["test"].dataset):.0f}%)\n')

# Train and test the model
if __name__ == '__main__':
    for epoch in range(1, 11):
        train(epoch)
        test()
    torch.save(model.state_dict(), 'model_font.pt')
