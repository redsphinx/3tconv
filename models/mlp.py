import torch
from torch.nn.functional import relu
from torch.nn import BatchNorm2d, BatchNorm3d, Conv3d, Conv2d
import torch.nn.functional as F

class MLP_basic(torch.nn.Module):
    def __init__(self, ksize, t_out, k_in_ch):
        super(MLP_basic, self).__init__()

        self.conv1 = Conv3d(in_channels=3, out_channels=1, kernel_size=(1, 7, 9), bias=False)
        self.bn1 = BatchNorm3d(1)

        self.conv2 = Conv2d(in_channels=k_in_ch, out_channels=1, kernel_size=ksize)
        self.bn1 = BatchNorm2d(1)

        self.fc1 = torch.nn.Linear(in_features=31, out_features=10)
        self.fc2 = torch.nn.Linear(in_features=10, out_features=10)

        self.fc_s = torch.nn.Linear(in_features=10, out_features=t_out)
        self.fc_r = torch.nn.Linear(in_features=10, out_features=t_out)
        self.fc_x = torch.nn.Linear(in_features=10, out_features=t_out)
        self.fc_y = torch.nn.Linear(in_features=10, out_features=t_out)

    def forward(self, input_, k0):
        reduced_input = F.interpolate(input_, (3, 30, 7, 9), mode='bilinear', align_corners=True)

        h1 = self.conv1(reduced_input)
        h1 = self.bn1(h1)

        h2 = self.conv2(k0)
        h2 = self.bn2(h2)

        h = torch.cat((h1, h2), 1) # TODO: concat in channel dimension

        h = self.fc1(h)
        h = relu(h)
        h = self.fc2(h)
        h = relu(h)

        s = self.fc_s(h)
        r = self.fc_r(h)
        x = self.fc_x(h)
        y = self.fc_y(h)

        return s, r, x, y
