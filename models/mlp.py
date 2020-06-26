import torch
from torch.nn.functional import relu
from torch.nn import BatchNorm2d, BatchNorm3d, Conv3d, Conv2d, AvgPool3d
import torch.nn.functional as F

class MLP_basic(torch.nn.Module):
    def __init__(self, ksize, t_out, k_in_ch):
        super(MLP_basic, self).__init__()

        self.conv1 = Conv3d(in_channels=3, out_channels=1, kernel_size=(1, 7, 9), bias=False)
        self.bn1 = BatchNorm3d(1)

        self.conv2 = Conv2d(in_channels=k_in_ch, out_channels=1, kernel_size=ksize)

        self.fc1 = torch.nn.Linear(in_features=31, out_features=10)
        self.fc2 = torch.nn.Linear(in_features=10, out_features=10)

        self.fc_s = torch.nn.Linear(in_features=10, out_features=t_out)
        self.fc_r = torch.nn.Linear(in_features=10, out_features=t_out)
        self.fc_x = torch.nn.Linear(in_features=10, out_features=t_out)
        self.fc_y = torch.nn.Linear(in_features=10, out_features=t_out)

    def forward(self, data, k0):
        # reduced_input = F.interpolate(input_, (30, 7, 9))
        # reduced_input = F.interpolate(input_, (30, 7, 9), mode='trilinear', align_corners=True)

        h1 = self.conv1(data)
        h1 = self.bn1(h1)

        h2 = self.conv2(k0)
        h2 = h2.unsqueeze(0)
        h2 = h2.repeat(h1.shape[0], 1, 1, 1, 1)

        h = torch.cat((h1, h2), 2)

        _shape = h.shape
        h = h.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])

        h = self.fc1(h)
        h = relu(h)
        h = self.fc2(h)
        h = relu(h)

        s = self.fc_s(h)[0]
        r = self.fc_r(h)[0]
        x = self.fc_x(h)[0]
        y = self.fc_y(h)[0]

        return s, r, x, y


class MLP_per_channel(torch.nn.Module):
    def __init__(self, in_channel, ksize, t_out, fc_in, hw):
        super(MLP_per_channel, self).__init__()

        # self.pool = AdaptiveAvgPool3d(output_size=(in_channel, temporal_dim, 1, 1))
        self.pool = AvgPool3d(kernel_size=(1, hw[0], hw[1]))




        # self.conv1 = Conv3d(in_channels=in_channel, out_channels=1, kernel_size=(ksize[1], 1, 1),
        #                     bias=False)

        self.fc1 = torch.nn.Linear(in_features=fc_in, out_features=10)


        self.fc_s = torch.nn.Linear(in_features=10, out_features=t_out)
        self.fc_r = torch.nn.Linear(in_features=10, out_features=t_out)
        self.fc_x = torch.nn.Linear(in_features=10, out_features=t_out)
        self.fc_y = torch.nn.Linear(in_features=10, out_features=t_out)

    def calc_difference(self, data, stride):


        data = F.pad(input=data, pad=[0, 0, 0, 0, 0, stride], mode='constant', value=0) - \
               F.pad(input=data, pad=[0, 0, 0, 0, stride, 0], mode='constant', value=0)

        data = torch.abs(data)[:, :, stride:-stride]
        data = F.avg_pool3d(data, (data.shape[2], 1, 1))



        return data


    def forward(self, data, tminone):

        h = self.pool(data)

        h1 = self.calc_difference(h, stride=1)

        for i in range(1, tminone):
            h2 = self.calc_difference(h, stride=i+1)
            h1 = torch.cat((h1, h2), dim=1)

        # h = self.conv1(h)

        _shape = h1.shape
        h = h1.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])

        h = self.fc1(h)
        h = relu(h)

        s = self.fc_s(h)[0]
        r = self.fc_r(h)[0]
        x = self.fc_x(h)[0]
        y = self.fc_y(h)[0]

        return s, r, x, y
