import torch
from torch.nn.functional import relu
from torch.nn import BatchNorm3d, AvgPool3d, Linear


from models.conv3t import ConvTTN3d


class ConvNet(torch.nn.Module):
    def __init__(self, pv):
        super(ConvNet, self).__init__()
        self.conv1 = ConvTTN3d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0, project_variable=pv, bias=False)
        self.pool1 = AvgPool3d(kernel_size=2)

        self.conv2 = ConvTTN3d(in_channels=16, out_channels=20, kernel_size=3, stride=1, padding=0, project_variable=pv, bias=False)
        self.conv3 = ConvTTN3d(in_channels=20, out_channels=32, kernel_size=5, stride=1, padding=0, project_variable=pv, bias=False)
        self.conv4 = ConvTTN3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0, project_variable=pv, bias=False)
        self.pool2 = AvgPool3d(kernel_size=2)

        self.fc1 = Linear(6, 1968)
        self.fc2 = Linear(1968, pv.label_size)



    def forward(self, x, device, stop_at=None, resized_datapoint=None):
        # h = self.conv1_relu(x, device)
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

        h = self.fc1(h)
        y = self.fc2(h)
