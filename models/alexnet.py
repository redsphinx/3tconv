import torch
from torch.nn.functional import relu, dropout3d
from torch.nn import MaxPool3d, AdaptiveAvgPool3d, Conv3d, BatchNorm3d, Linear


from models.conv3t import ConvTTN3d
from models.classic_conv3t import ConvTTN3d as classic_3TConv

# 53
class AlexNetExplicit3T(torch.nn.Module):
    def __init__(self, pv):
        super(AlexNetExplicit3T, self).__init__()
        self.conv1 = classic_3TConv(in_channels=3, out_channels=64, kernel_size=11, stride=(1, 3, 4), padding=2, project_variable=pv, bias=False)
        self.pool1 = MaxPool3d(kernel_size=3, stride=2)

        self.conv2 = classic_3TConv(in_channels=64, out_channels=192, kernel_size=5, padding=2, project_variable=pv, bias=False)
        self.pool2 = MaxPool3d(kernel_size=3, stride=2)

        self.conv3 = classic_3TConv(in_channels=192, out_channels=384, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.conv4 = classic_3TConv(in_channels=384, out_channels=256, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.conv5 = classic_3TConv(in_channels=256, out_channels=256, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.pool3 = MaxPool3d(kernel_size=3, stride=2)

        # self.pool4 = AdaptiveAvgPool3d(output_size=1)
        self.pool4 = AdaptiveAvgPool3d((1, 6, 6))

        self.fc1 = Linear(256 * 1 * 6 * 6, 4096)
        self.fc2 = Linear(4096, 4096)
        self.fc3 = Linear(4096, pv.label_size)


    def forward(self, x, device):
        # print('1. ', x.shape)
        h = self.conv1(x, device)
        # print('2. ', h.shape)
        h = relu(h)
        h = self.pool1(h)
        # print('3. ', h.shape)

        h = self.conv2(h, device)
        # print('4. ', h.shape)
        h = relu(h)
        h = self.pool2(h)
        # print('5. ', h.shape)

        h = self.conv3(h, device)
        # print('6. ', h.shape)
        h = relu(h)
        h = self.conv4(h, device)
        # print('7. ', h.shape)
        h = relu(h)
        h = self.conv5(h, device)
        # print('8. ', h.shape)
        h = relu(h)
        h = self.pool3(h)
        # print('9. ', h.shape)

        h = self.pool4(h)
        # print('10. ', h.shape)
        _shape = h.shape
        h = h.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        # print('11. ', h.shape)

        h = dropout3d(h, p=0.2)
        h = self.fc1(h)
        # print('12. ', h.shape)
        h = relu(h)
        h = dropout3d(h, p=0.2)
        h = self.fc2(h)
        # print('13. ', h.shape)
        h = relu(h)
        y = self.fc3(h)
        # print('14. ', h.shape)
        return y


# 54
class AlexNetExplicitTaco(torch.nn.Module):
    def __init__(self, pv):
        super(AlexNetExplicitTaco, self).__init__()
        self.conv1 = ConvTTN3d(in_channels=3, out_channels=64, kernel_size=11, stride=(1, 3, 4), padding=2, project_variable=pv, bias=False,
                               ksize=(0, 0), fc_in=1, hw=(150, 224))
        self.pool1 = MaxPool3d(kernel_size=3, stride=2)

        self.conv2 = ConvTTN3d(in_channels=64, out_channels=192, kernel_size=5, padding=2, project_variable=pv, bias=False,
                               ksize=(0, 0), fc_in=1, hw=(23, 27))
        self.pool2 = MaxPool3d(kernel_size=3, stride=2)

        self.conv3 = ConvTTN3d(in_channels=192, out_channels=384, kernel_size=3, padding=1, project_variable=pv, bias=False,
                               ksize=(0, 0), fc_in=1, hw=(11, 13))
        self.conv4 = ConvTTN3d(in_channels=384, out_channels=256, kernel_size=3, padding=1, project_variable=pv, bias=False,
                               ksize=(0, 0), fc_in=1, hw=(11, 13))
        self.conv5 = ConvTTN3d(in_channels=256, out_channels=256, kernel_size=3, padding=1, project_variable=pv, bias=False,
                               ksize=(0, 0), fc_in=1, hw=(11, 13))
        self.pool3 = MaxPool3d(kernel_size=3, stride=2)

        # self.pool4 = AdaptiveAvgPool3d(output_size=1)
        self.pool4 = AdaptiveAvgPool3d((1, 6, 6))

        self.fc1 = Linear(256 * 1 * 6 * 6, 4096)
        self.fc2 = Linear(4096, 4096)
        self.fc3 = Linear(4096, pv.label_size)


    def forward(self, x, device):
        print('1. ', x.shape)
        h = self.conv1(x, device)
        print('2. ', h.shape)
        h = relu(h)
        h = self.pool1(h)
        print('3. ', h.shape)

        h = self.conv2(h, device)
        print('4. ', h.shape)
        h = relu(h)
        h = self.pool2(h)
        print('5. ', h.shape)

        h = self.conv3(h, device)
        print('6. ', h.shape)
        h = relu(h)
        h = self.conv4(h, device)
        print('7. ', h.shape)
        h = relu(h)
        h = self.conv5(h, device)
        print('8. ', h.shape)
        h = relu(h)
        h = self.pool3(h)
        print('9. ', h.shape)

        h = self.pool4(h)
        print('10. ', h.shape)
        _shape = h.shape
        h = h.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        print('11. ', h.shape)

        h = dropout3d(h, p=0.2)
        h = self.fc1(h)
        print('12. ', h.shape)
        h = relu(h)
        h = dropout3d(h, p=0.2)
        h = self.fc2(h)
        print('13. ', h.shape)
        h = relu(h)
        y = self.fc3(h)
        print('14. ', h.shape)
        return y