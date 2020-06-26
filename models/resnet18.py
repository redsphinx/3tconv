import torch
from torch.nn.functional import relu
from torch.nn import MaxPool3d, AdaptiveAvgPool3d, Conv3d, BatchNorm3d

from models.conv3t import ConvTTN3d


class ResNet18Explicit(torch.nn.Module):
    def __init__(self, pv):
        super(ResNet18Explicit, self).__init__()
        # self.conv1_relu = ConvolutionBlock(3, 64, pv)
        self.conv1 = ConvTTN3d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, project_variable=pv, bias=False)
        self.bn1 = BatchNorm3d(64)

        self.maxpool = MaxPool3d(kernel_size=3, padding=1, stride=2, dilation=1)

        # self.res2a_relu = ResidualBlock(64, 64, pv)
        self.conv2 = ConvTTN3d(in_channels=64, out_channels=64, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.bn2 = BatchNorm3d(64)
        self.conv3 = ConvTTN3d(in_channels=64, out_channels=64, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.bn3 = BatchNorm3d(64)

        # self.res2b_relu = ResidualBlock(64, 64, pv)
        self.conv4 = ConvTTN3d(in_channels=64, out_channels=64, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.bn4 = BatchNorm3d(64)
        self.conv5 = ConvTTN3d(in_channels=64, out_channels=64, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.bn5 = BatchNorm3d(64)

        # self.res3a_relu = ResidualBlockB(64, 128, pv)
        self.conv6 = Conv3d(in_channels=64, out_channels=128, kernel_size=1, stride=2, bias=False)
        self.bn6 = BatchNorm3d(128)
        self.conv7 = ConvTTN3d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, project_variable=pv, bias=False)
        self.bn7 = BatchNorm3d(128)
        self.conv8 = ConvTTN3d(in_channels=128, out_channels=128, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.bn8 = BatchNorm3d(128)

        # self.res3b_relu = ResidualBlock(128, 128, pv)
        self.conv9 = ConvTTN3d(in_channels=128, out_channels=128, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.bn9 = BatchNorm3d(128)
        self.conv10 = ConvTTN3d(in_channels=128, out_channels=128, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.bn10 = BatchNorm3d(128)

        # self.res4a_relu = ResidualBlockB(128, 256, pv)
        self.conv11 = Conv3d(in_channels=128, out_channels=256, kernel_size=1, stride=2, bias=False)
        self.bn11 = BatchNorm3d(256)
        self.conv12 = ConvTTN3d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, project_variable=pv, bias=False)
        self.bn12 = BatchNorm3d(256)
        self.conv13 = ConvTTN3d(in_channels=256, out_channels=256, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.bn13 = BatchNorm3d(256)

        # self.res4b_relu = ResidualBlock(256, 256, pv)
        self.conv14 = ConvTTN3d(in_channels=256, out_channels=256, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.bn14 = BatchNorm3d(256)
        self.conv15 = ConvTTN3d(in_channels=256, out_channels=256, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.bn15 = BatchNorm3d(256)

        # self.res5a_relu = ResidualBlockB(256, 512, pv)
        self.conv16 = Conv3d(in_channels=256, out_channels=512, kernel_size=1, stride=2, bias=False)
        self.bn16 = BatchNorm3d(512)
        self.conv17 = ConvTTN3d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, project_variable=pv, bias=False)
        self.bn17 = BatchNorm3d(512)
        self.conv18 = ConvTTN3d(in_channels=512, out_channels=512, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.bn18 = BatchNorm3d(512)

        # self.res5b_relu = ResidualBlock(512, 512, pv)
        self.conv19 = ConvTTN3d(in_channels=512, out_channels=512, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.bn19 = BatchNorm3d(512)
        self.conv20 = ConvTTN3d(in_channels=512, out_channels=512, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.bn20 = BatchNorm3d(512)

        self.avgpool = AdaptiveAvgPool3d(output_size=1)
        self.fc = torch.nn.Linear(512, pv.label_size)

    def forward(self, x, device, stop_at=None):
        # h = self.conv1_relu(x, device)
        print('1: ', x.shape)
        num = 1
        h = self.conv1(x, device)
        if stop_at == num:
            return h
        h = self.bn1(h)
        h = relu(h)

        h = self.maxpool(h)

        print('2: ', h.shape)
        num = 2
        h1 = self.conv2(h, device)
        if stop_at == num:
            return h1
        h1 = self.bn2(h1)
        h1 = relu(h1)

        print('3: ', h1.shape)
        h1 = self.conv3(h1, device)
        if stop_at == num:
            return h1
        h1 = self.bn3(h1)
        h = h1 + h
        h = relu(h)

        print('4: ', h.shape)
        num = 4
        h1 = self.conv4(h, device)
        if stop_at == num:
            return h1
        h1 = self.bn4(h1)
        h1 = relu(h1)

        print('5: ', h1.shape)
        num = 5
        h1 = self.conv5(h1, device)
        if stop_at == num:
            return h1
        h1 = self.bn5(h1)
        h = h1 + h
        h = relu(h)

        # h = self.res3a_relu(h, device)
        temp = self.conv6(h)
        temp = self.bn6(temp)

        print('7: ', h.shape)
        num = 7
        h1 = self.conv7(h, device)
        if stop_at == num:
            return h1
        h1 = self.bn7(h1)
        h1 = relu(h1)
        print('8: ', h1.shape)
        num = 8
        h1 = self.conv8(h1, device)
        if stop_at == num:
            return h1
        h1 = self.bn8(h1)
        h = temp + h1
        h = relu(h)

        print('9: ', h.shape)
        num = 9
        h1 = self.conv9(h, device)
        if stop_at == num:
            return h1
        h1 = self.bn9(h1)
        h1 = relu(h1)
        print('10: ', h1.shape)
        num = 10
        h1 = self.conv10(h1, device)
        if stop_at == num:
            return h1
        h1 = self.bn10(h1)
        h = h1 + h
        h = relu(h)

        # h = self.res4a_relu(h, device)
        temp = self.conv11(h)
        temp = self.bn11(temp)
        print('12: ', h.shape)
        num = 12
        h1 = self.conv12(h, device)
        if stop_at == num:
            return h1
        h1 = self.bn12(h1)
        h1 = relu(h1)
        print('13: ', h1.shape)
        num = 13
        h1 = self.conv13(h1, device)
        if stop_at == num:
            return h1
        h1 = self.bn13(h1)
        h = temp + h1
        h = relu(h)

        print('14: ', h.shape)
        num = 14
        h1 = self.conv14(h, device)
        if stop_at == num:
            return h1
        h1 = self.bn14(h1)
        h1 = relu(h1)
        print('15: ', h1.shape)
        num = 15
        h1 = self.conv15(h1, device)
        if stop_at == num:
            return h1
        h1 = self.bn15(h1)
        h = h1 + h
        h = relu(h)

        temp = self.conv16(h)
        temp = self.bn16(temp)
        print('17: ', h.shape)
        num = 17
        h1 = self.conv17(h, device)
        if stop_at == num:
            return h1
        h1 = self.bn17(h1)
        h1 = relu(h1)
        print('18: ', h1.shape)
        num = 18
        h1 = self.conv18(h1, device)
        if stop_at == num:
            return h1
        h1 = self.bn18(h1)
        h = temp + h1
        h = relu(h)

        print('19: ', h.shape)
        num = 19
        h1 = self.conv19(h, device)
        if stop_at == num:
            return h1
        h1 = self.bn19(h1)
        h1 = relu(h1)
        print('20: ', h1.shape)
        num = 20
        h1 = self.conv20(h1, device)
        if stop_at == num:
            return h1
        h1 = self.bn20(h1)
        h = h1 + h
        h = relu(h)

        h = self.avgpool(h)
        _shape = h.shape
        h = h.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        y = self.fc(h)
        return y


class ResNet18Explicit3DConv(torch.nn.Module):
    def __init__(self, pv):
        super(ResNet18Explicit3DConv, self).__init__()
        # self.conv1_relu = ConvolutionBlock(3, 64, pv)
        self.conv1 = Conv3d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm3d(64)

        self.maxpool = MaxPool3d(kernel_size=3, padding=1, stride=2, dilation=1)

        # self.res2a_relu = ResidualBlock(64, 64, pv)
        self.conv2 = Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm3d(64)
        self.conv3 = Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.bn3 = BatchNorm3d(64)

        # self.res2b_relu = ResidualBlock(64, 64, pv)
        self.conv4 = Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.bn4 = BatchNorm3d(64)
        self.conv5 = Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.bn5 = BatchNorm3d(64)

        # self.res3a_relu = ResidualBlockB(64, 128, pv)
        self.conv6 = Conv3d(in_channels=64, out_channels=128, kernel_size=1, stride=2, bias=False)
        self.bn6 = BatchNorm3d(128)
        self.conv7 = Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn7 = BatchNorm3d(128)
        self.conv8 = Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False)
        self.bn8 = BatchNorm3d(128)

        # self.res3b_relu = ResidualBlock(128, 128, pv)
        self.conv9 = Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False)
        self.bn9 = BatchNorm3d(128)
        self.conv10 = Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False)
        self.bn10 = BatchNorm3d(128)

        # self.res4a_relu = ResidualBlockB(128, 256, pv)
        self.conv11 = Conv3d(in_channels=128, out_channels=256, kernel_size=1, stride=2, bias=False)
        self.bn11 = BatchNorm3d(256)
        self.conv12 = Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn12 = BatchNorm3d(256)
        self.conv13 = Conv3d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False)
        self.bn13 = BatchNorm3d(256)

        # self.res4b_relu = ResidualBlock(256, 256, pv)
        self.conv14 = Conv3d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False)
        self.bn14 = BatchNorm3d(256)
        self.conv15 = Conv3d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False)
        self.bn15 = BatchNorm3d(256)

        # self.res5a_relu = ResidualBlockB(256, 512, pv)
        self.conv16 = Conv3d(in_channels=256, out_channels=512, kernel_size=1, stride=2, bias=False)
        self.bn16 = BatchNorm3d(512)
        self.conv17 = Conv3d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn17 = BatchNorm3d(512)
        self.conv18 = Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False)
        self.bn18 = BatchNorm3d(512)

        # self.res5b_relu = ResidualBlock(512, 512, pv)
        self.conv19 = Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False)
        self.bn19 = BatchNorm3d(512)
        self.conv20 = Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False)
        self.bn20 = BatchNorm3d(512)

        self.avgpool = AdaptiveAvgPool3d(output_size=1)
        self.fc = torch.nn.Linear(512, pv.label_size)


    def forward(self, x, stop_at=None):
        # h = self.conv1_relu(x, device)

        num = 1
        h = self.conv1(x)
        if stop_at == num:
            return h
        h = self.bn1(h)
        h = relu(h)

        h = self.maxpool(h)

        # h = self.res2a_relu(h)
        num = 2
        h1 = self.conv2(h)
        if stop_at == num:
            return h1
        h1 = self.bn2(h1)
        h1 = relu(h1)
        num = 3
        h1 = self.conv3(h1)
        if stop_at == num:
            return h1
        h1 = self.bn3(h1)
        h = h1 + h
        h = relu(h)

        # h = self.res2b_relu(h)
        num = 4
        h1 = self.conv4(h)
        if stop_at == num:
            return h1
        h1 = self.bn4(h1)
        h1 = relu(h1)
        num = 5
        h1 = self.conv5(h1)
        if stop_at == num:
            return h1
        h1 = self.bn5(h1)
        h = h1 + h
        h = relu(h)

        # h = self.res3a_relu(h)
        temp = self.conv6(h)
        temp = self.bn6(temp)
        num = 7
        h1 = self.conv7(h)
        if stop_at == num:
            return h1
        h1 = self.bn7(h1)
        h1 = relu(h1)
        num = 8
        h1 = self.conv8(h1)
        if stop_at == num:
            return h1
        h1 = self.bn8(h1)
        h = temp + h1
        h = relu(h)

        # h = self.res3b_relu(h)
        num = 9
        h1 = self.conv9(h)
        if stop_at == num:
            return h1
        h1 = self.bn9(h1)
        h1 = relu(h1)
        num = 10
        h1 = self.conv10(h1)
        if stop_at == num:
            return h1
        h1 = self.bn10(h1)
        h = h1 + h
        h = relu(h)

        # h = self.res4a_relu(h)
        temp = self.conv11(h)
        temp = self.bn11(temp)
        num = 12
        h1 = self.conv12(h)
        if stop_at == num:
            return h1
        h1 = self.bn12(h1)
        h1 = relu(h1)
        num = 13
        h1 = self.conv13(h1)
        if stop_at == num:
            return h1
        h1 = self.bn13(h1)
        h = temp + h1
        h = relu(h)

        # h = self.res4b_relu(h)
        num = 14
        h1 = self.conv14(h)
        if stop_at == num:
            return h1
        h1 = self.bn14(h1)
        h1 = relu(h1)
        num = 15
        h1 = self.conv15(h1)
        if stop_at == num:
            return h1
        h1 = self.bn15(h1)
        h = h1 + h
        h = relu(h)

        # h = self.res5a_relu(h)
        temp = self.conv16(h)
        temp = self.bn16(temp)
        num = 17
        h1 = self.conv17(h)
        if stop_at == num:
            return h1
        h1 = self.bn17(h1)
        h1 = relu(h1)
        num = 18
        h1 = self.conv18(h1)
        if stop_at == num:
            return h1
        h1 = self.bn18(h1)
        h = temp + h1
        h = relu(h)

        # h = self.res5b_relu(h)
        num = 19
        h1 = self.conv19(h)
        if stop_at == num:
            return h1
        h1 = self.bn19(h1)
        h1 = relu(h1)
        num = 20
        h1 = self.conv20(h1)
        if stop_at == num:
            return h1
        h1 = self.bn20(h1)
        h = h1 + h
        h = relu(h)

        h = self.avgpool(h)
        _shape = h.shape
        h = h.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        y = self.fc(h)
        return y
    
    
# model 50 

class ResNet18ExplicitNiN(torch.nn.Module):
    def __init__(self, pv):
        super(ResNet18ExplicitNiN, self).__init__()
        # self.conv1_relu = ConvolutionBlock(3, 64, pv)
        self.conv1 = ConvTTN3d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, project_variable=pv,
                               bias=False, ksize=(3, 30), fc_in=1, hw=(150, 224))
        self.bn1 = BatchNorm3d(64)

        self.maxpool = MaxPool3d(kernel_size=3, padding=1, stride=2, dilation=1)

        # self.res2a_relu = ResidualBlock(64, 64, pv)
        self.conv2 = ConvTTN3d(in_channels=64, out_channels=64, kernel_size=3, padding=1, project_variable=pv,
                               bias=False, ksize=(64, 8), fc_in=1, hw=(38, 56))
        self.bn2 = BatchNorm3d(64)
        self.conv3 = ConvTTN3d(in_channels=64, out_channels=64, kernel_size=3, padding=1, project_variable=pv,
                               bias=False, ksize=(64, 8), fc_in=1, hw=(38, 56))
        self.bn3 = BatchNorm3d(64)

        # self.res2b_relu = ResidualBlock(64, 64, pv)
        self.conv4 = ConvTTN3d(in_channels=64, out_channels=64, kernel_size=3, padding=1, project_variable=pv,
                               bias=False, ksize=(64, 8), fc_in=1, hw=(38, 56))
        self.bn4 = BatchNorm3d(64)
        self.conv5 = ConvTTN3d(in_channels=64, out_channels=64, kernel_size=3, padding=1, project_variable=pv,
                               bias=False, ksize=(64, 8), fc_in=1, hw=(38, 56))
        self.bn5 = BatchNorm3d(64)

        # self.res3a_relu = ResidualBlockB(64, 128, pv)
        self.conv6 = Conv3d(in_channels=64, out_channels=128, kernel_size=1, stride=2, bias=False)
        self.bn6 = BatchNorm3d(128)
        self.conv7 = ConvTTN3d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1,
                               project_variable=pv, bias=False, ksize=(64, 8), fc_in=1, hw=(38, 56))
        self.bn7 = BatchNorm3d(128)
        self.conv8 = ConvTTN3d(in_channels=128, out_channels=128, kernel_size=3, padding=1, project_variable=pv,
                               bias=False, ksize=(128, 4), fc_in=1, hw=(19, 28))
        self.bn8 = BatchNorm3d(128)

        # self.res3b_relu = ResidualBlock(128, 128, pv)
        self.conv9 = ConvTTN3d(in_channels=128, out_channels=128, kernel_size=3, padding=1, project_variable=pv,
                               bias=False, ksize=(128, 4), fc_in=1, hw=(19, 28))
        self.bn9 = BatchNorm3d(128)
        self.conv10 = ConvTTN3d(in_channels=128, out_channels=128, kernel_size=3, padding=1, project_variable=pv,
                                bias=False, ksize=(128, 4), fc_in=1, hw=(19, 28))
        self.bn10 = BatchNorm3d(128)

        # self.res4a_relu = ResidualBlockB(128, 256, pv)
        self.conv11 = Conv3d(in_channels=128, out_channels=256, kernel_size=1, stride=2, bias=False)
        self.bn11 = BatchNorm3d(256)
        self.conv12 = ConvTTN3d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1,
                                project_variable=pv, bias=False, ksize=(128, 4), fc_in=1, hw=(19, 28))
        self.bn12 = BatchNorm3d(256)
        self.conv13 = ConvTTN3d(in_channels=256, out_channels=256, kernel_size=3, padding=1, project_variable=pv,
                                bias=False, ksize=(256, 2), fc_in=1, hw=(10, 14))
        self.bn13 = BatchNorm3d(256)

        # self.res4b_relu = ResidualBlock(256, 256, pv)
        self.conv14 = ConvTTN3d(in_channels=256, out_channels=256, kernel_size=3, padding=1, project_variable=pv,
                                bias=False, ksize=(256, 2), fc_in=1, hw=(10, 14))
        self.bn14 = BatchNorm3d(256)
        self.conv15 = ConvTTN3d(in_channels=256, out_channels=256, kernel_size=3, padding=1, project_variable=pv,
                                bias=False, ksize=(256, 2), fc_in=1, hw=(10, 14))
        self.bn15 = BatchNorm3d(256)

        # self.res5a_relu = ResidualBlockB(256, 512, pv)
        self.conv16 = Conv3d(in_channels=256, out_channels=512, kernel_size=1, stride=2, bias=False)
        self.bn16 = BatchNorm3d(512)
        self.conv17 = ConvTTN3d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1,
                                project_variable=pv, bias=False, ksize=(256, 2), fc_in=1, hw=(10, 14))
        self.bn17 = BatchNorm3d(512)
        self.conv18 = ConvTTN3d(in_channels=512, out_channels=512, kernel_size=3, padding=1, project_variable=pv,
                                bias=False, ksize=(512, 1), fc_in=1, hw=(5, 7))
        self.bn18 = BatchNorm3d(512)

        # self.res5b_relu = ResidualBlock(512, 512, pv)
        self.conv19 = ConvTTN3d(in_channels=512, out_channels=512, kernel_size=3, padding=1, project_variable=pv,
                                bias=False, ksize=(512, 1), fc_in=1, hw=(5, 7))
        self.bn19 = BatchNorm3d(512)
        self.conv20 = ConvTTN3d(in_channels=512, out_channels=512, kernel_size=3, padding=1, project_variable=pv,
                                bias=False, ksize=(512, 1), fc_in=1, hw=(5, 7))
        self.bn20 = BatchNorm3d(512)

        self.avgpool = AdaptiveAvgPool3d(output_size=1)
        self.fc = torch.nn.Linear(512, pv.label_size)

    def forward(self, x, device, stop_at=None, resized_datapoint=None):
        # h = self.conv1_relu(x, device)
        h = self.conv1(x, device, resized_datapoint)
        h = self.bn1(h)
        h = relu(h)

        h = self.maxpool(h)

        h1 = self.conv2(h, device)  # , resized_datapoint)
        h1 = self.bn2(h1)
        h1 = relu(h1)
        h1 = self.conv3(h1, device)  # , resized_datapoint)
        h1 = self.bn3(h1)
        h = h1 + h
        h = relu(h)

        # h = self.res2b_relu(h, device)  # , resized_datapoint)
        h1 = self.conv4(h, device)  # , resized_datapoint)
        h1 = self.bn4(h1)
        h1 = relu(h1)
        h1 = self.conv5(h1, device)  # , resized_datapoint)
        h1 = self.bn5(h1)
        h = h1 + h
        h = relu(h)

        # h = self.res3a_relu(h, device)  # , resized_datapoint)
        temp = self.conv6(h)
        temp = self.bn6(temp)
        h1 = self.conv7(h, device)  # , resized_datapoint)
        h1 = self.bn7(h1)
        h1 = relu(h1)
        h1 = self.conv8(h1, device)  # , resized_datapoint)
        h1 = self.bn8(h1)
        h = temp + h1
        h = relu(h)

        # h = self.res3b_relu(h, device)  # , resized_datapoint)
        h1 = self.conv9(h, device)  # , resized_datapoint)
        h1 = self.bn9(h1)
        h1 = relu(h1)
        h1 = self.conv10(h1, device)  # , resized_datapoint)
        h1 = self.bn10(h1)
        h = h1 + h
        h = relu(h)

        # h = self.res4a_relu(h, device)  # , resized_datapoint)
        temp = self.conv11(h)
        temp = self.bn11(temp)
        h1 = self.conv12(h, device)  # , resized_datapoint)
        h1 = self.bn12(h1)
        h1 = relu(h1)
        h1 = self.conv13(h1, device)  # , resized_datapoint)
        h1 = self.bn13(h1)
        h = temp + h1
        h = relu(h)

        # h = self.res4b_relu(h, device)  # , resized_datapoint)
        h1 = self.conv14(h, device)  # , resized_datapoint)
        h1 = self.bn14(h1)
        h1 = relu(h1)
        h1 = self.conv15(h1, device)  # , resized_datapoint)
        h1 = self.bn15(h1)
        h = h1 + h
        h = relu(h)

        # h = self.res5a_relu(h, device)  # , resized_datapoint)
        temp = self.conv16(h)
        temp = self.bn16(temp)
        h1 = self.conv17(h, device)  # , resized_datapoint)
        h1 = self.bn17(h1)
        h1 = relu(h1)
        h1 = self.conv18(h1, device)  # , resized_datapoint)
        h1 = self.bn18(h1)
        h = temp + h1
        h = relu(h)

        # h = self.res5b_relu(h, device)  # , resized_datapoint)
        h1 = self.conv19(h, device)  # , resized_datapoint)
        h1 = self.bn19(h1)
        h1 = relu(h1)
        h1 = self.conv20(h1, device)  # , resized_datapoint)
        h1 = self.bn20(h1)
        h = h1 + h
        h = relu(h)

        h = self.avgpool(h)
        _shape = h.shape
        h = h.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        y = self.fc(h)
        return y