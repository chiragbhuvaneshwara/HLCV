import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

import matplotlib.pyplot as plt

import os
n = 2
np.random.seed(n)
torch.cuda.manual_seed_all(n)
torch.manual_seed(n)

def weights_init(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight.data, mean=0.0, std=1e-3)
        m.bias.data.fill_(0.0)
        
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data) 
        m.bias.data.fill_(0.0)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

#--------------------------------
# Hyper-parameters
#--------------------------------
input_size = 3
num_classes = 10
hidden_size = [128, 512, 512, 512, 512, 512]
num_epochs = 20
batch_size = 200
learning_rate = 2e-3
learning_rate_decay = 0.95
reg=0.001
num_training= 49000
num_validation =1000
norm_layer = None
print(hidden_size)


#-------------------------------------------------
# Load the CIFAR-10 dataset
#-------------------------------------------------
#################################################################################
# TODO: Q3.a Chose the right data augmentation transforms with the right        #
# hyper-parameters and put them in the data_aug_transforms variable             #
#################################################################################
data_aug_transforms = []
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
norm_transform = transforms.Compose(data_aug_transforms+[transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])
test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])
cifar_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                           train=True,
                                           transform=norm_transform,
                                           download=False)

test_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                          train=False,
                                          transform=test_transform
                                          )
#-------------------------------------------------
# Prepare the training and validation splits
#-------------------------------------------------
mask = list(range(num_training))
train_dataset = torch.utils.data.Subset(cifar_dataset, mask)
mask = list(range(num_training, num_training + num_validation))
val_dataset = torch.utils.data.Subset(cifar_dataset, mask)

#-------------------------------------------------
# Data loader
#-------------------------------------------------
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

train_loader

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


#-------------------------------------------------
# Convolutional neural network (Q1.a and Q2.a)
# Set norm_layer for different networks whether using batch normalization
#-------------------------------------------------
class ConvNet(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, p, norm_layer=None):
        super(ConvNet, self).__init__()
        #################################################################################
        # TODO: Initialize the modules required to implement the convolutional layer    #
        # described in the exercise.                                                    #
        # For Q1.a make use of conv2d and relu layers from the torch.nn module.         #
        # For Q2.a make use of BatchNorm2d layer from the torch.nn module.              #
        # For Q3.b Use Dropout layer from the torch.nn module.                          #
        #################################################################################
        layers = []
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        
        self.conv = nn.Sequential(
                    nn.Conv2d(in_channels=input_size, out_channels=hidden_layers[0], kernel_size=3, stride=1, padding=1),
                    # nn.BatchNorm2d(hidden_layers[0]),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.ReLU(),
                    nn.Dropout2d(p),
        
                    nn.Conv2d(in_channels=hidden_layers[0], out_channels=hidden_layers[1], kernel_size=3, stride=1, padding=1),
                    # nn.BatchNorm2d(hidden_layers[1]),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.ReLU(),
                    nn.Dropout2d(p),
        
                    nn.Conv2d(in_channels=hidden_layers[1], out_channels=hidden_layers[2], kernel_size=3, stride=1, padding=1),
                    # nn.BatchNorm2d(hidden_layers[2]),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.ReLU(),
                    nn.Dropout2d(p),
        
                    nn.Conv2d(in_channels=hidden_layers[2], out_channels=hidden_layers[3], kernel_size=3, stride=1, padding=1),
                    # nn.BatchNorm2d(hidden_layers[3]),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.ReLU(),
                    nn.Dropout2d(p),
        
                    nn.Conv2d(in_channels=hidden_layers[3], out_channels=hidden_layers[4], kernel_size=3, stride=1, padding=1),
                    # nn.BatchNorm2d(hidden_layers[4]),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.ReLU(),
                    nn.Dropout2d(p)    )
        
        self.fc = nn.Sequential(
                    nn.Linear(hidden_layers[4], hidden_layers[5]),
                    # nn.BatchNorm1d(hidden_layers[5]),
                    nn.ReLU(),
        
                    nn.Linear(hidden_layers[5], num_classes)
                        )
        
        layers = [self.conv, self.fc]
        self.layers = nn.Sequential(*layers)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        x = self.conv(x)
        x = x.squeeze()
        out = self.fc(x)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out


#-------------------------------------------------
# Calculate the model size (Q1.b)
# if disp is true, print the model parameters, otherwise, only return the number of parameters.
#-------------------------------------------------
def PrintModelSize(model, disp=True):
    #################################################################################
    # TODO: Implement the function to count the number of trainable parameters in   #
    # the input model. This useful to track the capacity of the model you are       #
    # training                                                                      #
    #################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    model_sz = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model_sz)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return model_sz

#-------------------------------------------------
# Calculate the model size (Q1.c)
# visualize the convolution filters of the first convolution layer of the input model
#-------------------------------------------------
def VisualizeFilter(model):
    #################################################################################
    # TODO: Implement the functiont to visualize the weights in the first conv layer#
    # in the model. Visualize them as a single image fo stacked filters.            #
    # You can use matlplotlib.imshow to visualize an image in python                #
    #################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # print(type(model.layers[0].weight.data))

    
    tensor = torch.ones(model.layers[0][0].weight.data.size())
    # tensor.new_tensor(model.layers[0].weight.data, requires_grad=False)
    tensor = model.layers[0][0].weight.clone().detach().requires_grad_(False)
    tensor = tensor.cpu().data.numpy()
    

    t_max = np.amax(tensor)
    t_min = np.amin(tensor)

    tensor = (tensor - t_min) / (t_max - t_min) 

    num_cols = 16
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    fig.set_facecolor("black")

    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i],)
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#======================================================================================
# Q1.a: Implementing convolutional neural net in PyTorch
#======================================================================================
# In this question we will implement a convolutional neural networks using the PyTorch
# library.  Please complete the code for the ConvNet class evaluating the model
#--------------------------------------------------------------------------------------
pVals = [i/10 for i in range(1,10)]
pVals = [0.1, 0.2]

trainAcc = []
valAcc = []
for p in pVals:

    print("######################################################################")
    print("current value of p is:", p)
    print("######################################################################")

    model = ConvNet(input_size, hidden_size, num_classes, p, norm_layer=norm_layer).to(device)
    # Q2.a - Initialize the model with correct batch norm layer

    model.apply(weights_init)
    # Print the model
    # print(model)
    # Print model size
    #======================================================================================
    # Q1.b: Implementing the function to count the number of trainable parameters in the model
    #======================================================================================
    PrintModelSize(model)
    #======================================================================================
    # Q1.a: Implementing the function to visualize the filters in the first conv layers.
    # Visualize the filters before training
    #======================================================================================
    # VisualizeFilter(model)


    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)

    # Train the model
    lr = learning_rate
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        # Code to update the lr
        lr *= learning_rate_decay
        update_lr(optimizer, lr)
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Validataion accuracy is: {} %'.format(100 * correct / total))
            #################################################################################
            # TODO: Q2.b Implement the early stopping mechanism to save the model which has #
            # acheieved the best validation accuracy so-far.                                #
            #################################################################################
            best_model = None
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        model.train()
        
    trainAcc.append(100 * correct / total)


    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    model.eval()
    #################################################################################
    # TODO: Q2.b Implement the early stopping mechanism to load the weights from the#
    # best model so far and perform testing with this model.                        #
    #################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if total == 1000:
                break

        print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))
        valAcc.append(100 * correct / total)

    # Q1.c: Implementing the function to visualize the filters in the first conv layers.
    # Visualize the filters before training
    # VisualizeFilter(model)
    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')
# print(trainAcc, valAcc)
plt.plot(pVals, valAcc, label = "Test Acc")
plt.plot(pVals, trainAcc, label = "Train Acc")
plt.xlabel("Dropout Probability")
plt.ylabel("Accuracy")
plt.legend(loc="upper right")
plt.show()