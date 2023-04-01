import torch
from torch import nn
import numpy as np


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self,x):
        return x.view(x.size(0), -1)

class ConvStandard(nn.Conv2d): 
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0, w_sig =\
                 np.sqrt(1.0)):
        super(ConvStandard, self).__init__(in_channels, out_channels,kernel_size)
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.w_sig = w_sig
        self.reset_parameters()
      
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0, std=self.w_sig/(self.in_channels*np.prod(self.kernel_size)))
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0, std=0)
            
    def forward(self, input):
        return F.conv2d(input,self.weight,self.bias,self.stride,self.padding)
            
class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
#             model += [ConvStandard(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
#                                 )]
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=not batch_norm)]
        else:
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model)

class AllCNN(nn.Module):
    def __init__(self, filters_percentage=1., n_channels=3, num_classes=10, dropout=False, batch_norm=True):
        super(AllCNN, self).__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)
        
        self.conv1 = Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm)
        self.conv2 = Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm)
        self.conv3 = Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm)
        
        self.dropout1 = self.features = nn.Sequential(nn.Dropout(inplace=True) if dropout else Identity())
        
        self.conv4 = Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv5 = Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv6 = Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm)
        
        self.dropout2 = self.features = nn.Sequential(nn.Dropout(inplace=True) if dropout else Identity())
        
        self.conv7 = Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv8 = Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm)
        if n_channels == 3:
            self.pool = nn.AvgPool2d(8)
        elif n_channels == 1:
            self.pool = nn.AvgPool2d(7)
        self.flatten = Flatten()
        
        self.classifier = nn.Sequential(
            nn.Linear(n_filter2, num_classes),
        )

    def forward(self, x):
        out = self.conv1(x)
        actv1 = out
        
        out = self.conv2(out)
        actv2 = out
        
        out = self.conv3(out)
        actv3 = out
        
        out = self.dropout1(out)
        
        out = self.conv4(out)
        actv4 = out
        
        out = self.conv5(out)
        actv5 = out
        
        out = self.conv6(out)
        actv6 = out
        
        out = self.dropout2(out)
        
        out = self.conv7(out)
        actv7 = out
        
        out = self.conv8(out)
        actv8 = out
        
        out = self.pool(out)
        
        out = self.flatten(out)
        
        out = self.classifier(out)
        
        return out, actv1, actv2, actv3, actv4, actv5, actv6, actv7, actv8 
    
    
class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class LeNet32(nn.Module):
    def __init__(self, n_classes):
        super(LeNet32, self).__init__()
        self.n_classes = n_classes

        self.layers = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            View((-1, 16*5*5)),
            nn.Linear(16*5*5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, n_classes))


    def forward(self, x, true_labels=None):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx == 0:
                activation1 = x
            if idx == 3:
                activation2 = x

        return x, activation1, activation2
    

class ResidualBlock(nn.Module):
    """
    A residual block as defined by He et al.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ResidualBlock, self).__init__()
        self.conv_res1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, stride=stride, bias=False)
        self.conv_res1_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
        self.conv_res2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, bias=False)
        self.conv_res2_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)

        if stride != 1:
            # in case stride is not set to 1, we need to downsample the residual so that
            # the dimensions are the same when we add them together
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv_res1_bn(self.conv_res1(x)))
        out = self.conv_res2_bn(self.conv_res2(out))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.relu(out)
        out += residual
        return out


class ResNet9(nn.Module):
    """
    A Residual network.
    """
    def __init__(self):
        super(Net, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Linear(in_features=1024, out_features=10, bias=True)

    def forward(self, x):
        for idx, layer in enumerate(self.conv):
            x = layer(x)
            if idx == 0:
                activation1 = x
            if idx == 3:
                activation2 = x
            if idx == 8:
                activation3 = x
            if idx == 12:
                activation4 = x
        
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.fc(x)
        return x, activation1, activation2, activation3, activation4