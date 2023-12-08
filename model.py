import torch.nn as nn # PyTorch neural network module
import torch.nn.functional as F # PyTorch functional module
import torch.optim as optim # PyTorch optimization module

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