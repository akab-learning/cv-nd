## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # 1 input image channel (grayscale), 32 output channels/feature maps
        # 5x5 square convolution kernel
        # Input Image width = 224 
        
        # First conv layer
        ## output size = (W-F)/S +1 = (224-4)/1 +1 = 221
        # the output Tensor for one image, will have the dimensions: (32, 221, 221)
        # after one pool layer, this becomes (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 4)
        
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.drop1 = nn.Dropout(p=0.1)
        
        # Second conv layer
        ## output size = (W-F)/S +1 = (110-3)/1 +1 = 108
        # the output Tensor for one image, will have the dimensions: (64, 108, 108)
        # after one pool layer, this becomes (64, 54, 54)
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.drop2 = nn.Dropout(p=0.2)
        
        # Third conv layer
        ## output size = (W-F)/S +1 = (54-2)/1 +1 = 53
        # the output Tensor for one image, will have the dimensions: (128, 53, 53)
        # after one pool layer, this becomes (128, 26, 26)
        self.conv3 = nn.Conv2d(64, 128, 2)
        
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.drop3 = nn.Dropout(p=0.3)
        
        # Fourth conv layer
        ## output size = (W-F)/S +1 = (26-1)/1 +1 = 26
        # the output Tensor for one image, will have the dimensions: (256, 26, 26)
        # after one pool layer, this becomes (256, 13, 13)
        self.conv4 = nn.Conv2d(128, 256, 1)
        
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.drop4 = nn.Dropout(p=0.4)
        
        # 256 outputs * the 13*13 filtered/pooled map size
        self.fc1 = nn.Linear(256*13*13, 1024)
        
        # dropout with p=0.4
        self.drop5 = nn.Dropout(p=0.5)
        
        # finally, create 136 output channels (2 for each of the 68 keypoint (x, y) pairs)
        self.fc2 = nn.Linear(1024, 136)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        x = F.elu(self.drop1(self.pool1(self.conv1(x))))
        x = F.elu(self.drop2(self.pool2(self.conv2(x))))
        x = F.elu(self.drop3(self.pool3(self.conv3(x))))
        x = F.elu(self.drop4(self.pool4(self.conv4(x))))
        
        # prep for linear layer
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.elu(self.drop5(self.fc1(x)))
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
