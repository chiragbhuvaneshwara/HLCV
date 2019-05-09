import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sys

def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)

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
input_size = 32 * 32 * 3
# hidden_size = [50, 40, 30, 25, 20]
hidden_size = [50]
num_classes = 10
num_epochs = 10
batch_size = 200
learning_rate = 1e-3
learning_rate_decay = 0.95
reg=0.001
num_training= 49000
num_validation =1000
train = True
train = False

#-------------------------------------------------
# Load the CIFAR-10 dataset
#-------------------------------------------------
norm_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])
cifar_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                           train=True,
                                           transform=norm_transform,
                                           download=False)

test_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                          train=False,
                                          transform=norm_transform
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

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

#======================================================================================
# Q4: Implementing multi-layer perceptron in PyTorch
#======================================================================================
# So far we have implemented a two-layer network using numpy by explicitly
# writing down the forward computation and deriving and implementing the
# equations for backward computation. This process can be tedious to extend to
# large network architectures
#
# Popular deep-learining libraries like PyTorch and Tensorflow allow us to
# quickly implement complicated neural network architectures. They provide
# pre-defined layers which can be used as building blocks to define our
# network. They also enable automatic-differentiation, which allows us to
# define only the forward pass and let the libraries perform back-propagation
# using automatic differentiation.
#
# In this question we will implement a multi-layer perceptron using the PyTorch
# library.  Please complete the code for the MultiLayerPerceptron, training and
# evaluating the model. Once you can train the two layer model, experiment with
# adding more layers and
#--------------------------------------------------------------------------------------

#-------------------------------------------------
# Fully connected neural network with one hidden layer
#-------------------------------------------------
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes):
        super(MultiLayerPerceptron, self).__init__()
        #################################################################################
        # TODO: Initialize the modules required to implement the mlp with given layer   #
        # configuration. input_size --> hidden_layers[0] --> hidden_layers[1] .... -->  #
        # hidden_layers[-1] --> num_classes                                             #
        # Make use of linear and relu layers from the torch.nn module                   #
        #################################################################################
        layers = []
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append( nn.ReLU() )

        for i in range(1, len(hidden_layers)):
            
            layers.append( nn.Linear(hidden_layers[i-1], hidden_layers[i]) )
            layers.append( nn.ReLU() )
        
        layers.append( nn.Linear(hidden_layers[-1], num_classes) )
        
        # self.fc1 = nn.Linear(input_size, hidden_layers[0])
        
        # self.fc2 = nn.Linear(hidden_layers[0], num_classes)
        
        # # Define RELU activation and softmax output 
        # self.relu = nn.ReLU()
        # # self.softmax = nn.Softmax(dim=1)

        # layers = [self.fc1, self.relu, self.fc2] #, self.softmax]
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        # Note that you do not need to use the softmax operation at the end.            #
        # Softmax is only required for the loss computation and the criterion used below#
        # nn.CrossEntropyLoss() already integrates the softmax and the log loss together#
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # x = self.fc1(x)
        # x = self.relu(x)
        # out = self.fc2(x)

        # out = self.softmax(x)
        
        for i in range(len(self.layers)):
            x = self.layers[i](x)

        out = x
        # x = self.layers[](x)
        # x = self.relu(x)
        # out = self.fc2(x)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out

model = MultiLayerPerceptron(input_size, hidden_size, num_classes).to(device)
print(model)
if train:
    model.apply(weights_init)


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
            #################################################################################
            # TODO: Implement the training code                                             #
            # 1. Pass the images to the model                                               #
            # 2. Compute the loss using the output and the labels.                          #
            # 3. Compute gradients and update the model using the optimizer                 #
            # Use examples in https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
            #################################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            
            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(images.reshape(batch_size,-1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        # Code to update the lr
        lr *= learning_rate_decay
        update_lr(optimizer, lr)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                ####################################################
                # TODO: Implement the evaluation code              #
                # 1. Pass the images to the model                  #
                # 2. Get the most confident predicted class        #
                ####################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                predicted = model(images.reshape(batch_size,-1))
                _ , predicted = torch.max(predicted, 1)
                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Validataion accuracy is: {} %'.format(100 * correct / total))

    ##################################################################################
    # TODO: Now that you can train a simple two-layer MLP using above code, you can  #
    # easily experiment with adding more layers and different layer configurations   #
    # and let the pytorch library handle computing the gradients                     #
    #                                                                                #
    # Experiment with different number of layers (atleast from 2 to 5 layers) and    #
    # record the final validation accuracies Report your observations on how adding  #
    # more layers to the MLP affects its behavior. Try to improve the model          #
    # configuration using the validation performance as the guidance. You can        #
    # experiment with different activation layers available in torch.nn, adding      #
    # dropout layers, if you are interested. Use the best model on the validation    #
    # set, to evaluate the performance on the test set once and report it            #
    ##################################################################################

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')

else:
    # Run the test code once you have your by setting train flag to false
    # and loading the best model

    best_model = torch.load('model.ckpt')
    model.load_state_dict(best_model)
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            ####################################################
            # TODO: Implement the evaluation code              #
            # 1. Pass the images to the model                  #
            # 2. Get the most confident predicted class        #
            ####################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            outputs = model(images.reshape(batch_size,-1))
            _ , predicted = torch.max(outputs, 1)

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if total == 1000:
                break

        print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))

