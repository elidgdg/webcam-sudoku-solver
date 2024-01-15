from torch.utils.data import DataLoader # PyTorch data loader module
from torchvision import datasets # PyTorch vision module
from torchvision.transforms import ToTensor # Transform the data into PyTorch Tensors
import torch # PyTorch module
import torch.nn as nn # PyTorch neural network module
import torch.nn.functional as F # PyTorch functional module
import torch.optim as optim # PyTorch optimization module

# Download training data
training_data = datasets.MNIST(
    root="data", # data directory
    train=True, # training data
    transform=ToTensor(), # convert the samples into Tensors
    download=True # download from internet if not available at root
)

# Download test data
test_data = datasets.MNIST(
    root="data", # data directory
    train=False, # test data
    transform=ToTensor(), # convert the samples into Tensors
    download=True # download from internet if not available at root
)

# Define the data loaders
loaders = {
    'train': DataLoader(training_data, batch_size=100, shuffle=True, num_workers=1),
    'test': DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1)
}

# Define the neural network model class
class Net(nn.Module): 
    def __init__(self):
        super(Net, self).__init__() # Call the parent class constructor

        # Define the layers of the neural network
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) #First convolutional layer
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) #Second convolutional layer
        self.conv2_drop = nn.Dropout2d() # Dropout layer
        self.fc1 = nn.Linear(320, 50) # First fully connected layer
        self.fc2 = nn.Linear(50, 10) # Second fully connected layer

    # Define the forward pass through the network
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # Apply the first convolutional layer, ReLU activation, and max pooling
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) # Apply the second convolutional layer, ReLU activation, dropout, and max pooling
        x = x.view(-1, 320) # Reshape the output of the convolutional layers to be a vector
        x = F.relu(self.fc1(x)) # Apply the first fully connected layer and ReLU activation
        x = F.dropout(x, training=self.training) # Apply dropout for the fully connected layer
        x = self.fc2(x) # Apply the second fully connected layer
        return F.softmax(x) # Apply softmax activation to the output to get probabilities
    
# Create the model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Check if a GPU is available
model = Net().to(device) # Create the model and send it to the GPU if available
optimizer = optim.Adam(model.parameters(), lr=0.001) # Create the optimizer
loss_fn = nn.CrossEntropyLoss() # Create the loss function

# Define the training function
def train(epoch):
    model.train() # Set the model to training mode
    
    for batch_idx, (data, target) in enumerate(loaders['train']): # Iterate over the training data
        data, target = data.to(device), target.to(device) # Send the data to the GPU if available
        optimizer.zero_grad() # Zero the gradients
        output = model(data) # Forward pass through the network
        loss = loss_fn(output, target) # Calculate the loss
        loss.backward() # Backpropagate the loss
        optimizer.step() # Update the weights
        if batch_idx % 20 == 0: # Print the loss every 20 batches
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)} / {len(loaders["train"].dataset)} ({100. * batch_idx / len(loaders["train"]):.0f}%)]\tLoss: {loss.item():.6f}')

# Define the test function
def test():
    model.eval() # Set the model to evaluation mode
    test_loss = 0 
    correct = 0

    with torch.no_grad(): # Disable gradient calculation
        for data, target in loaders['test']: # Iterate over the test data
            data, target = data.to(device), target.to(device) # Send the data to the GPU if available
            output = model(data) # Forward pass through the network
            test_loss += loss_fn(output, target).item() # Calculate the loss
            pred = output.argmax(dim=1, keepdim=True) # Get the predicted class
            correct += pred.eq(target.view_as(pred)).sum().item() # Calculate the number of correct predictions
    test_loss /= len(loaders['test'].dataset) # Calculate the average loss

    # Print the test results
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loaders["test"].dataset)} ({100. * correct / len(loaders["test"].dataset):.0f}%)\n')

# Train and test the model
if __name__ == '__main__': # If this file is being run directly
    for epoch in range(1, 11):
        train(epoch)
        test()
    torch.save(model.state_dict(), 'model.pt') # Save the model parameters to a file 