import torch
from torch.nn.functional import relu
from torch.nn import BatchNorm3d, AvgPool3d, Linear

from models.conv3t import ConvTTN3d
from models.classic_conv3t import ConvTTN3d as classic_3tconv


# 51
class ConvNet3T(torch.nn.Module):
    def __init__(self, pv):
        super(ConvNet3T, self).__init__()
        self.conv1 = classic_3tconv(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0, project_variable=pv, bias=False)
        self.pool1 = AvgPool3d(kernel_size=2)

        self.conv2 = classic_3tconv(in_channels=16, out_channels=20, kernel_size=5, stride=1, padding=0, project_variable=pv, bias=False)
        self.conv3 = classic_3tconv(in_channels=20, out_channels=32, kernel_size=5, stride=1, padding=0, project_variable=pv, bias=False)
        self.conv4 = classic_3tconv(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=0, project_variable=pv, bias=False)
        self.pool2 = AvgPool3d(kernel_size=2)

        if pv.dataset == 'jester':
            features_in = 48608
        elif pv.dataset == 'ucf101':
            features_in = 54880
        else:
            print('ERROR: Dataset not valid, features_in cannot be set')
            features_in = None

        self.fc1 = Linear(features_in, 1968)
        self.fc2 = Linear(1968, pv.label_size)


    def forward(self, x, device, stop_at=None):
        # print('1. ', x.shape)
        h = self.conv1(x, device)
        h = relu(h)
        h = self.pool1(h)

        # print('2. ', h.shape)
        h = self.conv2(h, device)

        h = relu(h)
        # print('3. ', h.shape)
        h = self.conv3(h, device)
        h = relu(h)
        # print('4. ', h.shape)
        h = self.conv4(h, device)
        h = relu(h)
        # print('5. ', h.shape)
        h = self.pool2(h)

        _shape = h.shape
        h = h.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])

        h = self.fc1(h)
        h = relu(h)
        y = self.fc2(h)
        return y

# 52
class TACoNet(torch.nn.Module):
    def __init__(self, pv):
        super(TACoNet, self).__init__()
        # self.conv1 = ConvTTN3d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0, project_variable=pv, bias=False)
        self.conv1 = ConvTTN3d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0, project_variable=pv,
                               bias=False, ksize=None, fc_in=1, hw=(150, 224))
        self.pool1 = AvgPool3d(kernel_size=2)

        # self.conv2 = ConvTTN3d(in_channels=16, out_channels=20, kernel_size=5, stride=1, padding=0, project_variable=pv, bias=False)
        self.conv2 = ConvTTN3d(in_channels=16, out_channels=20, kernel_size=5, stride=1, padding=0, project_variable=pv,
                               bias=False, ksize=None, fc_in=1, hw=(74, 111))

        # self.conv3 = ConvTTN3d(in_channels=20, out_channels=32, kernel_size=5, stride=1, padding=0, project_variable=pv, bias=False)
        self.conv3 = ConvTTN3d(in_channels=20, out_channels=32, kernel_size=5, stride=1, padding=0, project_variable=pv,
                               bias=False, ksize=None, fc_in=1, hw=(70, 107))

        # self.conv4 = ConvTTN3d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=0, project_variable=pv, bias=False)
        self.conv4 = ConvTTN3d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=0, project_variable=pv,
                               bias=False, ksize=None, fc_in=1, hw=(66, 103))

        self.pool2 = AvgPool3d(kernel_size=2)

        if pv.dataset == 'jester':
            features_in = 48608
        elif pv.dataset == 'ucf101':
            features_in = 54880
        else:
            print('ERROR: Dataset not valid, features_in cannot be set')
            features_in = None

        self.fc1 = Linear(features_in, 1968)
        self.fc2 = Linear(1968, pv.label_size)


    def forward(self, x, device, stop_at=None, resized_datapoint=None):
        h = self.conv1(x, device, resized_datapoint)
        h = relu(h)
        h = self.pool1(h)

        h = self.conv2(h, device)
        h = relu(h)
        h = self.conv3(h, device)
        h = relu(h)
        h = self.conv4(h, device)
        h = relu(h)
        h = self.pool2(h)

        _shape = h.shape
        h = h.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])

        h = self.fc1(h)
        h = relu(h)
        y = self.fc2(h)
        return y
