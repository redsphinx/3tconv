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

kernel_size = (1, 5, 5)

filter = np.zeros((1, 1, kernel_size[1], kernel_size[2]), dtype=float)
filter[0, 0] = np.array([[0, 0, 0, 0, 0], [0, 1, 0, -1, 0], [0, 2, 0, -2, 0], [0, 1, 0, -1, 0], [0, 0, 0, 0, 0]])
filter = torch.from_numpy(filter)

image_size = (1, 100, 100)
image = None

scale = None
rotate = None
translate_x = None
translate_y = None


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
    # deal with updating s r x y

    # if theta is not None: (theta is not None because we use srxy starting from eye)
    
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
                              [out_channels, something[0] - 1, something[1],
                               something[2]])
        else:
            _ = F.affine_grid(theta[0],
                              [out_channels, something[0]-1, something[1],
                               something[2]], align_corners=True)
    except RuntimeError:
        torch.backends.cudnn.deterministic = True
        print('ok cudnn')

    if something[0] > 1:
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

    return grid
    


def params_x_something(mult_w_filter, mult_w_image):
    if mult_w_filter:
        something_size = kernel_size
        something = filter
    elif mult_w_image:
        something_size = image_size
        something = image
    else:
        print('ERROR: something is not defined')
        something_size = None
        something = None

    theta = torch.zeros((1, out_channels, 2, 3))
    theta = theta.cuda(device)

    grid = torch.zeros((1, out_channels, something_size[1], something_size[2], 2))
    grid = grid.cuda(device)
    grid = make_grid(grid, theta, something_size)
    grid = grid[1:]

    try:
        if torch.__version__ == '1.2.0':
            _ = F.grid_sample(something[:, :, 0], grid[0], mode='bilinear', padding_mode='zeros')
        else:
            _ = F.grid_sample(something[:, :, 0], grid[0], mode='bilinear', padding_mode='zeros',
                              align_corners=True)
    except RuntimeError:
        torch.backends.cudnn.deterministic = True
        print('ok cudnn')
        # ---

    if torch.__version__ == '1.2.0':
        result = F.grid_sample(something[:, :, -1], grid, mode='bilinear', padding_mode='zeros')
    else:
        result = F.grid_sample(something[:, :, -1], grid, mode='bilinear', padding_mode='zeros',
                               align_corners=True)


    return result


params_x_something(mult_w_filter=True, mult_w_image=False)