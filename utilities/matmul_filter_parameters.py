from torch.nn import functional as F
import numpy as np
import torch


'''

compare:
results1 = (params x filter) * input
results2 = params x (filter * input)

'''



device = None
out_channels = 1
in_channels = 1

kernel_size = (1, 3, 3)

filter = np.zeros((1, 1, kernel_size[1], kernel_size[2]), dtype=float)
filter[0, 0] = np.array([[-1, 0, 2],
                         [-2, 0, 3],
                         [-1, 0, 2]])
filter = torch.from_numpy(filter)


image = np.array([[5,6,7,8,9],
                 [4,8,1,2,1],
                 [4,5,6,7,8],
                 [1,2,3,4,5]])


scale = torch.from_numpy(np.array([0.5], dtype=float)).unsqueeze(0)
rotate = torch.from_numpy(np.array([0], dtype=float)).unsqueeze(0)
translate_x = torch.from_numpy(np.array([0], dtype=float)).unsqueeze(0)
translate_y = torch.from_numpy(np.array([0], dtype=float)).unsqueeze(0)


def make_affine_matrix(scale, rotate, translate_x, translate_y):
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


def make_grid(grid, theta, something):
    if something[0] > 1:
        for i in range(something[0]-1):
            tmp = make_affine_matrix(scale[i], rotate[i], translate_x[i], translate_y[i])
            tmp = tmp.cuda(device)
            theta = torch.cat((theta, tmp.unsqueeze(0)), 0)
        theta = theta[1:]
    else:
        tmp = make_affine_matrix(scale[0], rotate[0], translate_x[0], translate_y[0])
        tmp = tmp.cuda(device)
        theta = torch.cat((theta, tmp.unsqueeze(0)), 0)
        theta = theta[1:]


    try:
        if torch.__version__ == '1.2.0':
            _ = F.affine_grid(theta[0],
                              [out_channels, something[0], something[1],
                               something[2]])
        else:
            _ = F.affine_grid(theta[0],
                              [out_channels, something[0], something[1], something[2]], align_corners=True)
    except RuntimeError:
        torch.backends.cudnn.deterministic = True
        print('ok cudnn')

    if something[0] < 2:
        if torch.__version__ == '1.2.0':
            tmp = F.affine_grid(theta[0],
                                [out_channels, something[0], something[1],
                                 something[2]])
        else:
            tmp = F.affine_grid(theta[0],
                                [out_channels, something[0], something[1],
                                 something[2]], align_corners=True)

        grid = torch.cat((grid, tmp.unsqueeze(0)), 0)

    else:
        for i in range(something[0] - 1):
            if torch.__version__ == '1.2.0':
                tmp = F.affine_grid(theta[i],
                                    [out_channels, something[0], something[1],
                                     something[2]])
            else:
                tmp = F.affine_grid(theta[i],
                                    [out_channels, something[0], something[1],
                                     something[2]], align_corners=True)

            grid = torch.cat((grid, tmp.unsqueeze(0)), 0)

    return grid, theta
    

def params_x_something(mult_w_filter, mult_w_image, which_image):

    if mult_w_filter:
        something_size = kernel_size
        something = filter
    elif mult_w_image:
        _im = torch.from_numpy(which_image)
        _im = _im.unsqueeze(0)
        something_size = _im.shape
        _im = _im.unsqueeze(0)
        something = _im

    else:
        print('ERROR: something is not defined')
        something_size = None
        something = None

    something = something.cuda(device)
    something = something.type(torch.float32)

    theta = torch.zeros((1, out_channels, 2, 3))
    theta = theta.cuda(device)

    grid = torch.zeros((1, out_channels, something_size[1], something_size[2], 2))
    grid = grid.cuda(device)
    grid, new_theta = make_grid(grid, theta, something_size)
    grid = grid[1:]

    try:
        if torch.__version__ == '1.2.0':
            _ = F.grid_sample(something[:, :, 0], grid[0], mode='bilinear', padding_mode='zeros')
        else:
            _ = F.grid_sample(something, grid[0], mode='bilinear', padding_mode='zeros', align_corners=True)
    except RuntimeError:
        torch.backends.cudnn.deterministic = True
        print('ok cudnn')
        # ---

    if torch.__version__ == '1.2.0':
        result = F.grid_sample(something[:, :, -1], grid, mode='bilinear', padding_mode='zeros')
    else:
        result = F.grid_sample(something, grid[0], mode='bilinear', padding_mode='zeros', align_corners=True)

    print('params x something')
    print(something)
    print(new_theta)
    print(result)

    return result


def filter_x_image(which_filter):
    _im = torch.from_numpy(image)
    _im = _im.unsqueeze(0)
    _im = _im.unsqueeze(0)
    the_image = _im.cuda(device)

    the_filter = which_filter.cuda(device)

    the_image = the_image.type(torch.float32)
    the_filter = the_filter.type(torch.float32)

    result_conv2d = F.conv2d(the_image, the_filter, stride=1)

    result1 = np.array(result_conv2d[0][0].cpu())

    print('filter x input')
    print(the_image)
    print(the_filter)
    print(result1)

    return result1

'''
compare:
results1 = (params x filter) * input
results2 = params x (filter * input)
------------

CONCLUSION
(params x filter) * input != params x (filter * input)

------------
'''

## result1 = (params x filter) * input
# new_filter = params_x_something(mult_w_filter=True, mult_w_image=False, which_image=None)
# img = filter_x_image(new_filter)

'''
params x something
tensor([[[[-1.,  0.,  2.],
          [-2.,  0.,  3.],
          [-1.,  0.,  2.]]]], device='cuda:0')
tensor([[[[0.5000, -0.0000, 0.0000],
          [0.0000, 0.5000, 0.0000]]]], device='cuda:0')
tensor([[[[-0.7500,  0.0000,  1.2500],
          [-1.0000,  0.0000,  1.5000],
          [-0.7500,  0.0000,  1.2500]]]], device='cuda:0')
          
filter x input
tensor([[[[5., 6., 7., 8., 9.],
          [4., 8., 1., 2., 1.],
          [4., 5., 6., 7., 8.],
          [1., 2., 3., 4., 5.]]]], device='cuda:0')
tensor([[[[-0.7500,  0.0000,  1.2500],
          [-1.0000,  0.0000,  1.5000],
          [-0.7500,  0.0000,  1.2500]]]], device='cuda:0')
[[ 7.    5.5  12.  ]
 [ 6.25  5.5  10.5 ]]
'''


## result2 = params x (filter * input)
# new_image = filter_x_image(filter)
# ft = params_x_something(mult_w_filter=False, mult_w_image=True, which_image=new_image)
'''
filter x input
tensor([[[[5., 6., 7., 8., 9.],
          [4., 8., 1., 2., 1.],
          [4., 5., 6., 7., 8.],
          [1., 2., 3., 4., 5.]]]], device='cuda:0')
tensor([[[[-1.,  0.,  2.],
          [-2.,  0.,  3.],
          [-1.,  0.,  2.]]]], device='cuda:0')
[[12.  9. 22.]
 [13. 13. 20.]]
 
params x something
tensor([[[[12.,  9., 22.],
          [13., 13., 20.]]]], device='cuda:0')
tensor([[[[0.5000, -0.0000, 0.0000],
          [0.0000, 0.5000, 0.0000]]]], device='cuda:0')
tensor([[[[11.1250, 10.0000, 15.7500],
          [12.3750, 12.0000, 16.2500]]]], device='cuda:0')
'''


'''
input:
[[5,6,7,8,9],
 [4,8,1,2,1],
 [4,5,6,7,8],
 [1,2,3,4,5]]

filter:
[[-1, 0, 2],
 [-2, 0, 3],
 [-1, 0, 2]]
 
params:
[[0.5, 0, 0],
 [0, 0.5, 0]] 

result1 = (params x filter) * input
[[ 7.    5.5  12.  ]
 [ 6.25  5.5  10.5 ]]
 
result2 = params x (filter * input)
[[11.125, 10., 15.75],
 [12.375, 12., 16.25]]
'''
