import torch
from torch.nn.functional import relu

from models.classic_conv3t import ConvTTN3d as classic_3TConv


# 55
class LeNet5_2d(torch.nn.Module):

    def __init__(self, project_variable):
        super(LeNet5_2d, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        # Max-pooling
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        # Convolution
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        # Max-pooling
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)
        # Fully connected layer
        self.fc1 = torch.nn.Linear(16*6*6,
                                   120)  # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
        self.fc2 = torch.nn.Linear(120, 84)  # convert matrix with 120 features to a matrix of 84 features (columns)
        self.fc3 = torch.nn.Linear(84, project_variable.label_size)  # convert matrix with 84 features to a matrix of 10 features (columns)

    def forward(self, x):
        # convolve, then perform ReLU non-linearity
        x = relu(self.conv1(x))
        # max-pooling with 2x2 grid
        x = self.max_pool_1(x)
        # convolve, then perform ReLU non-linearity
        x = relu(self.conv2(x))
        # max-pooling with 2x2 grid
        x = self.max_pool_2(x)
        # first flatten 'max_pool_2_out' to contain 16*5*5 columns
        # read through https://stackoverflow.com/a/42482819/7551231
        x = x.view(-1, 16 * 6 * 6)
        # FC-1, then perform ReLU non-linearity
        x = relu(self.fc1(x))
        # FC-2, then perform ReLU non-linearity
        x = relu(self.fc2(x))
        # FC-3
        x = self.fc3(x)

        return x


# 56
class LeNet5_3t(torch.nn.Module):
    def __init__(self, pv):
        super(LeNet5_3t, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
        self.conv1 = classic_3TConv(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True,
                                    project_variable=pv)
        # Max-pooling
        self.max_pool_1 = torch.nn.MaxPool3d(kernel_size=2)
        # Convolution
        self.conv2 = classic_3TConv(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True,
                                    project_variable=pv)
        # Max-pooling
        self.max_pool_2 = torch.nn.MaxPool3d(kernel_size=2)
        # Fully connected layer
        self.fc1 = torch.nn.Linear(16 * 6 * 6,
                                   120)  # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
        self.fc2 = torch.nn.Linear(120, 84)  # convert matrix with 120 features to a matrix of 84 features (columns)
        self.fc3 = torch.nn.Linear(84, pv.label_size)  # convert matrix with 84 features to a matrix of 10 features (columns)


    def forward(self, x, device):
        x = self.conv1(x, device)
        x = relu(x)
        x = self.max_pool_1(x)
        x = self.conv2(x, device)
        x = relu(x)
        x = self.max_pool_2(x)
        _shape = x.shape
        x = x.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        # _shape = x.shape
        # x = x.view(-1, _shape[1] * 5 * 5 * 5)
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        x = relu(x)
        x = self.fc3(x)
        return x