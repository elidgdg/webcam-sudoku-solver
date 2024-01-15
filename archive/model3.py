from torch.utils.data import Dataset, DataLoader # PyTorch data loader module
from torchvision import datasets # PyTorch vision module
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize # Transform the data into PyTorch Tensors
import torch # PyTorch module
import torch.nn as nn # PyTorch neural network module
import torch.nn.functional as F # PyTorch functional module
import torch.optim as optim # PyTorch optimization module
import os # Operating system module
from PIL import Image # Python image library


class Chars74KDataset(Dataset):
    def __init__(self, root_folder, transform=None):
        self.root_folder = root_folder
        self.transform = transform
        self.data = []
        self.targets = []
        
        for digit in range(10):
            digit_folder = os.path.join(self.root_folder, str(digit))
            for filename in os.listdir(digit_folder):
                img_path = os.path.join(digit_folder, filename)
                self.data.append(img_path)
                self.targets.append(digit)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path, target = self.data[index], self.targets[index]
        img = Image.open(img_path).convert("L")

        if self.transform:
            img = self.transform(img)

        return img, target

root_folder_chars74k = "chars74k_dataset"
chars74k_dataset = Chars74KDataset(root_folder_chars74k, transform=transforms.Compose([transforms.ToTensor(), Resize((28, 28))]))

mnist_dataset = datasets.MNIST(
    root="data", # data directory
    train=True, # training data
    transform=ToTensor(), # convert the samples into Tensors
    download=True # download from internet if not available at root
)

combined_dataset = chars74k_dataset + mnist_dataset
train_loader = DataLoader(combined_dataset, batch_size=100, shuffle=True, num_workers=1)

class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__() # Call the parent class constructor
        self.fc1 = nn.Linear(28*28, 128) # First fully connected layer  
        self.fc2 = nn.Linear(128, 10) # Second fully connected layer

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x)) # Apply the first fully connected layer and ReLU activation
        x = self.fc2(x) # Apply the second fully connected layer
        return F.log_softmax(x, dim=1) # Apply softmax activation to the output to get probabilities

model = CombinedModel()
optimizer = optim.Adam(model.parameters(), lr=0.001) # Create the optimizer
loss_fn = nn.CrossEntropyLoss() # Create the loss function

def train(epoch):
    model.train() # Set the model to training mode
    
    for batch_idx, (data, target) in enumerate(train_loader): # Iterate over the training data
        # data, target = data.to(device), target.to(device) # Send the data to the GPU if available
        optimizer.zero_grad() # Zero the gradients
        output = model(data) # Forward pass through the network
        loss = loss_fn(output, target) # Calculate the loss
        loss.backward() # Backpropagate the loss
        optimizer.step() # Update the weights
        if batch_idx % 20 == 0: # Print the loss every 20 batches
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)} / {len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# def test():
#     model.eval() # Set the model to evaluation mode
#     test_loss = 0 
#     correct = 0

#     with torch.no_grad(): # Disable gradient calculation


#     # Print the test results
#     print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loaders["test"].dataset)} ({100. * correct / len(loaders["test"].dataset):.0f}%)\n')


if __name__ == "__main__":
    for epoch in range(1, 10):
        train(epoch)
        # test()
    torch.save(model.state_dict(), "combined_model.pt")