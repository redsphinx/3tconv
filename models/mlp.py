import torch
from torch.nn.functional import relu
from torch.nn import MaxPool3d, AdaptiveAvgPool3d, Conv3d, BatchNorm3d


class MLP_basic(torch.nn.Module):
    def __init__(self, inp_feat, t_out):
        super(MLP_basic, self).__init__()

        self.fc1 = torch.nn.Linear(in_features=inp_feat[0], out_features=10)
        self.bn1 = BatchNorm3d(10)
        self.fc2 = torch.nn.Linear(in_features=10, out_features=5)
        self.bn2 = BatchNorm3d(5)
        self.fc3 = torch.nn.Linear(in_features=5, out_features=t_out)

    def forward(self, x):
        h = self.fc1(x)
        h = self.bn1(h)
        h = relu(h)
        h = self.fc2(h)
        h = self.bn2(h)
        h = relu(h)
        y = self.fc3(h)

        return y
