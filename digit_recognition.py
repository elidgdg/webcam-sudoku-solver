from torchvision import datasets
from torchvision.transforms import ToTensor

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