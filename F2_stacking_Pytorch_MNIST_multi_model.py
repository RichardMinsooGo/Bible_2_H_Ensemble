'''
Data Engineering
'''

'''
D1. Import Libraries for Data Engineering
'''
import numpy as np

import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

'''
D2. Load MNIST data
'''
train_dataset = datasets.MNIST(root='./data', 
                             download=True,
                             train=True,
                             transform=transforms.ToTensor())

test_dataset = datasets.MNIST(root='./data', 
                            download=True,
                            train=False,
                             transform=transforms.ToTensor())

'''
Model Engineering
'''

'''
M1. Import Libraries for Model Engineering
'''

import torch.nn as nn
import torch.optim as optimizers

np.random.seed(123)
torch.manual_seed(123)

'''
M2. Set Hyperparameters
'''
hidden_size = 256 
output_dim = 10 # output layer dimensionality = num_classes
EPOCHS = 30
batch_size = 100
learning_rate = 0.001

'''
M3. DataLoader
'''

train_ds = torch.utils.data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size, 
                                       shuffle=True)
test_ds = torch.utils.data.DataLoader(dataset=test_dataset,
                                      batch_size=batch_size, 
                                      shuffle=False)

'''
M4. Build NN model
'''
# 3-Layers Convolution neural network with one hidden layer
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        
        # Convolution 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        
        # Max pool 1
        self.maxpool2d1 = nn.MaxPool2d(kernel_size=2)
     
        # Convolution 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        
        # Max pool 2
        self.maxpool2d2 = nn.MaxPool2d(kernel_size=2)
        
        # Convolution 3
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        
        # Max pool 3
        self.maxpool2d3 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected 1 (readout)
        self.d1 = nn.Linear(256 * 1 * 1, 256) 
        self.d1_2 = nn.Linear(128 * 5 * 5, 256) 
        self.d1_3 = nn.Linear(64 * 13 * 13, 256) 
        self.d2 = nn.Dropout(0.2)
        self.d3 = nn.Linear(256, 256) 
        self.d4 = nn.Linear(256*3, output_dim) 

    def forward(self, x):
        
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        x1 = self.maxpool2d1(x1)
        x1 = self.conv2(x1)
        x1 = self.relu2(x1)
        x1 = self.maxpool2d2(x1)
        x1 = self.conv3(x1)
        x1 = self.relu3(x1)
        x1 = self.maxpool2d3(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.d1(x1)
        x1 = self.d2(x1)
        x1 = self.d3(x1)
        
        x2 = self.conv1(x)
        x2 = self.relu1(x2)
        x2 = self.maxpool2d1(x2)
        x2 = self.conv2(x2)
        x2 = self.relu2(x2)
        x2 = self.maxpool2d3(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.d1_2(x2)
        x2 = self.d2(x2)
        x2 = self.d3(x2)

        x3 = self.conv1(x)
        x3 = self.relu1(x3)
        x3 = self.maxpool2d1(x3)
        x3 = x3.view(x3.size(0), -1)
        x3 = self.d1_3(x3)
        x3 = self.d2(x3)
        x3 = self.d3(x3)

        # Concatenate in dim1 (feature dimension)
        out = torch.cat((x1, x2, x3), 1)
        
        out = self.d4(out)
        
        return out

'''
M5. Transfer model to GPU
'''

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = CNN_Model(input_size, hidden_size, output_dim).to(device)
model = CNN_Model().to(device)

'''
M6. Optimizer
'''
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

'''
M7. Define Loss Fumction
'''

criterion = nn.CrossEntropyLoss()

'''
M8. Define train loop
'''

def train_step(model, images, labels):
    model.train()
    # origin shape: [100, 1, 28, 28]
    images = images.to(device)
    labels = labels.to(device)

    # Forward pass
    predictions = model(images)
    loss = criterion(predictions, labels)
    loss_val = loss.item()

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Pytorch need a manual coding for accuracy
    # max returns (value ,index)
    _, predicted = torch.max(predictions.data, 1)           
    n_samples = labels.size(0)
    n_correct = (predicted == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    
    return loss_val, acc

'''
M9. Define validation / test loop
'''

def test_step(model, images, labels):
    model.eval()
    # origin shape: [100, 1, 28, 28]
    images = images.to(device)
    labels = labels.to(device)

    # Forward pass
    predictions = model(images)
    loss = criterion(predictions, labels)
    loss_val = loss.item()

    # Pytorch need a manual coding for accuracy
    # max returns (value ,index)
    _, predicted = torch.max(predictions.data, 1)           
    n_samples = labels.size(0)
    n_correct = (predicted == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    
    return loss_val, acc

'''
M10. Define Episode / each step process
'''
from tqdm import tqdm, tqdm_notebook, trange

for epoch in range(EPOCHS):
    
    with tqdm_notebook(total=len(train_ds), desc=f"Train Epoch {epoch+1}") as pbar:    
        train_losses = []
        train_accuracies = []
        
        for i, (images, labels) in enumerate(train_ds):
         
            loss_val, acc = train_step(model, images, labels)
            
            train_losses.append(loss_val)
            train_accuracies.append(acc)
            
            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.4f} ({np.mean(train_losses):.4f}) Acc: {acc:.3f} ({np.mean(train_accuracies):.3f})")


'''
M11. Model evaluation
'''
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():

    with tqdm_notebook(total=len(test_ds), desc=f"Test_ Epoch {epoch+1}") as pbar:    
        test_losses = []
        test_accuracies = []

        for images, labels in test_ds:
            loss_val, acc = test_step(model, images, labels)

            test_losses.append(loss_val)
            test_accuracies.append(acc)

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.4f} ({np.mean(test_losses):.4f}) Acc: {acc:.3f} ({np.mean(test_accuracies):.3f})")
            
