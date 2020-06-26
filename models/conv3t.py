import torch
from torch.nn.modules import conv
from torch.nn import functional as F, ModuleList
from torch.nn.modules.utils import _triple

from models.mlp import MLP_basic, MLP_per_channel

class ConvTTN3d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, project_variable, transformation_groups=None,
                 k0_groups=None, transformations_per_filter=None, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', ksize=None, fc_in=None, hw=None):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        super(ConvTTN3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias, padding_mode)

        self.project_variable = project_variable

        # replaced with k0_groups with out_channels
        first_w = torch.nn.Parameter(torch.zeros(self.out_channels, in_channels, 1, kernel_size[1], kernel_size[2]))

        # default: k0_init = 'normal'
        self.first_weight = torch.nn.init.normal_(first_w)

        if project_variable.nin:


            self.all_mlps = ModuleList([MLP_per_channel(in_channels, ksize, kernel_size[0]-1, fc_in, hw) for i in range(out_channels)])

            # self.mlp = MLP_basic(ksize=(kernel_size[1], kernel_size[2]), t_out=kernel_size[0]-1, k_in_ch=in_channels)

            # this is a tensor, NOT a parameter
            # time x channels
            _time = self.kernel_size[0]-1
            _ch = self.out_channels
            self.scale = torch.zeros((_time, 1))  # , _ch))
            self.rotate = torch.zeros((_time, 1))  # , _ch))
            self.translate_x = torch.zeros((_time, 1))  # , _ch))
            self.translate_y = torch.zeros((_time, 1))  # , _ch))

            self.og_scale = torch.zeros((_time, 1))
            self.og_rotate = torch.zeros((_time, 1))
            self.og_translate_x = torch.zeros((_time, 1))
            self.og_translate_y = torch.zeros((_time, 1))
            
        else:
            self.scale = torch.nn.Parameter(
                torch.nn.init.ones_(torch.zeros((self.kernel_size[0]-1, self.out_channels))))
            self.rotate = torch.nn.Parameter(
                torch.zeros((self.kernel_size[0]-1, self.out_channels)))
            self.translate_x = torch.nn.Parameter(
                torch.zeros((self.kernel_size[0]-1, self.out_channels)))
            self.translate_y = torch.nn.Parameter(
                torch.zeros((self.kernel_size[0]-1, self.out_channels)))

    def make_affine_matrix(self, scale, rotate, translate_x, translate_y):
        # if out_channels is used, the shape of the matrix returned is different

        assert scale.shape == rotate.shape == translate_x.shape == translate_y.shape

        '''
        matrix.shape = (out_channels, 2, 3)
        '''
        matrix = torch.zeros((scale.shape[0], 2, 3))

        matrix[:, 0, 0] = scale[:] * torch.cos(rotate[:])
        matrix[:, 0, 1] = -scale[:] * torch.sin(rotate[:])
        matrix[:, 0, 2] = translate_x[:] * scale[:] * torch.cos(rotate[:]) - translate_y[:] * \
                          scale[:] * torch.sin(rotate[:])
        matrix[:, 1, 0] = scale[:] * torch.sin(rotate[:])
        matrix[:, 1, 1] = scale[:] * torch.cos(rotate[:])
        matrix[:, 1, 2] = translate_x[:] * scale[:] * torch.sin(rotate[:]) + translate_y[:] * \
                          scale[:] * torch.cos(rotate[:])

        return matrix


    def update_2(self, grid, theta, device):
        # deal with updating s r x y

        # if theta is not None: (theta is not None because we use srxy starting from eye)

        for i in range(self.kernel_size[0]-1):
            tmp = self.make_affine_matrix(self.scale[i], self.rotate[i], self.translate_x[i], self.translate_y[i])
            tmp = tmp.cuda(device)
            if theta[0].shape != tmp.shape:
                print('not match')
            theta = torch.cat((theta, tmp.unsqueeze(0)), 0)
        theta = theta[1:]

        try:
            if torch.__version__ == '1.2.0':
                _ = F.affine_grid(theta[0],
                                  [self.out_channels, self.kernel_size[0] - 1, self.kernel_size[1],
                                   self.kernel_size[2]])
            else:
                _ = F.affine_grid(theta[0],
                                  [self.out_channels, self.kernel_size[0]-1, self.kernel_size[1],
                                   self.kernel_size[2]], align_corners=True)
        except RuntimeError:
            torch.backends.cudnn.deterministic = True
            print('ok cudnn')

        for i in range(self.kernel_size[0] - 1):
            if torch.__version__ == '1.2.0':
                tmp = F.affine_grid(theta[i],
                                    [self.out_channels, self.kernel_size[0], self.kernel_size[1],
                                     self.kernel_size[2]])
            else:
                tmp = F.affine_grid(theta[i],
                                    [self.out_channels, self.kernel_size[0], self.kernel_size[1],
                                     self.kernel_size[2]], align_corners=True)

            grid = torch.cat((grid, tmp.unsqueeze(0)), 0)

        return grid



    def generate_srxy(self, datapoint, device): #resized_datapoint):
        self.scale = self.scale.cuda(device)
        self.rotate = self.rotate.cuda(device)
        self.translate_x = self.translate_x.cuda(device)
        self.translate_y = self.translate_y.cuda(device)
        
        for i in range(self.out_channels):

            # self.scale[:, i],self.rotate[:, i],self.translate_x[:, i],self.translate_y[:, i] = self.all_mlps[i](datapoint)
            
            _tmp = self.all_mlps[i](datapoint)

            # # _tmp = self.mlp(resized_datapoint, self.first_weight[i, :, 0].unsqueeze(0))
            #
            # for t in range(self.kernel_size[0]-1):
            self.scale = torch.cat((_tmp[0].unsqueeze(1), self.scale), 1)
            self.rotate = torch.cat((_tmp[1].unsqueeze(1), self.rotate), 1)
            self.translate_x = torch.cat((_tmp[2].unsqueeze(1), self.translate_x), 1)
            self.translate_y = torch.cat((_tmp[3].unsqueeze(1), self.translate_y), 1)
                
        self.scale = self.scale[:, 1:]
        self.rotate = self.rotate[:, 1:]
        self.translate_x = self.translate_x[:, 1:]
        self.translate_y = self.translate_y[:, 1:]
            
                

                # self.scale[t, i] = _tmp[0][t]
                # self.rotate[t, i] = _tmp[1][t]
                # self.translate_x[t, i] = _tmp[2][t]
                # self.translate_y[t, i] = _tmp[3][t]

        # for i in range(self.out_channels):
        #     _tmp = self.mlp(resized_datapoint, self.first_weight[i, :, 0].unsqueeze(0))
        #
        #     for t in range(self.kernel_size[0]-1):
        #         self.scale[t, i] = _tmp[0][t]
        #         self.rotate[t, i] = _tmp[1][t]
        #         self.translate_x[t, i] = _tmp[2][t]
        #         self.translate_y[t, i] = _tmp[3][t]

    
    def reset_srxy(self):
        self.scale = self.og_scale.clone()
        self.rotate = self.og_rotate.clone()
        self.translate_x = self.og_translate_x.clone()
        self.translate_y = self.og_translate_y.clone()
        
    
    def forward(self, input_, device, resized_datapoint=None):
        
        if self.project_variable.nin:
            # average in h, w dimensions
            self.reset_srxy()
            self.generate_srxy(input_, device)

            # if resized_datapoint is None:
            #     self.generate_srxy(input_)
            # else:
            #     self.generate_srxy(resized_datapoint)
            
        grid = torch.zeros((1, self.out_channels, self.kernel_size[1], self.kernel_size[2], 2))

        theta = torch.zeros((1, self.out_channels, 2, 3))
        theta = theta.cuda(device)

        grid = grid.cuda(device)
        grid = self.update_2(grid, theta, device)
        grid = grid[1:]


        # ---
        # needed to deal with the cudnn error
        try:
            if torch.__version__ == '1.2.0':
                _ = F.grid_sample(self.first_weight[:, :, 0], grid[0], mode='bilinear', padding_mode='zeros')
            else:
                _ = F.grid_sample(self.first_weight[:, :, 0], grid[0], mode='bilinear', padding_mode='zeros',
                                  align_corners=True)
        except RuntimeError:
            torch.backends.cudnn.deterministic = True
            print('ok cudnn')
        # ---

        new_weight = self.first_weight

        # default: weight_transform = 'seq'
        for i in range(self.kernel_size[0] - 1):
            if torch.__version__ == '1.2.0':
                tmp = F.grid_sample(new_weight[:, :, -1], grid[i], mode='bilinear', padding_mode='zeros')
            else:
                tmp = F.grid_sample(new_weight[:, :, -1], grid[i], mode='bilinear', padding_mode='zeros',
                                    align_corners=True)
            new_weight = torch.cat((new_weight, tmp.unsqueeze(2)), 2)

        self.weight = torch.nn.Parameter(new_weight)

        y = F.conv3d(input_, new_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y