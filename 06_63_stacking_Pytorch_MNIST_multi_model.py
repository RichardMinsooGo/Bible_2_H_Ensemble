import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 784 # 28x28
hidden_size = 256 
num_classes = 10
EPOCHS = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_ds = torch.utils.data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size, 
                                       shuffle=True)

test_ds = torch.utils.data.DataLoader(dataset=test_dataset,
                                      batch_size=batch_size, 
                                      shuffle=False)

# plot first few images
examples = iter(test_ds)
example_data, example_targets = examples.next()

for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(example_data[i][0], cmap=plt.get_cmap('gray'))
    # if you want to invert color, you can use 'gray_r'. this can be used only for MNIST, Fashion MNIST not cifar10
    # pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray_r'))
    
# show the figure
plt.show()

# 3-Layers Convolution neural network with one hidden layer
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool2d1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2d2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        self.maxpool2d3 = nn.MaxPool2d(kernel_size=2)
        self.d1 = nn.Linear(256 * 1 * 1, 256) 
        self.d1_2 = nn.Linear(128 * 5 * 5, 256) 
        self.d1_3 = nn.Linear(64 * 13 * 13, 256) 
        self.d2 = nn.Dropout(0.2)
        self.d3 = nn.Linear(256, 256) 
        self.d4 = nn.Linear(256*3, num_classes) 

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
    
# model = CNN_Model(input_size, hidden_size, num_classes).to(device)
model = CNN_Model().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
n_total_steps = len(train_ds)

def train_step(model, images, labels):
    model.train()
    # origin shape: [100, 1, 28, 28]
    images = images.to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss_val = loss.item()

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Pytorch need a manual coding for accuracy
    # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)           
    n_samples = labels.size(0)
    n_correct = (predicted == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    
    return loss_val, acc

def test_step(model, images, labels):
    model.eval()
    # origin shape: [100, 1, 28, 28]
    images = images.to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss_val = loss.item()

    # Pytorch need a manual coding for accuracy
    # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)           
    n_samples = labels.size(0)
    n_correct = (predicted == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    
    return loss_val, acc

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


    # Test the model
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
            
