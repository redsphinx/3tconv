import os
from torch.nn import functional as F
import numpy as np
import torch
from PIL import Image


'''

compare:
results1 = (params x filter) * input
results2 = params x (filter * input)

'''

device = None
out_channels = 1
in_channels = 1
save_path = '/home/gabras/3tconv/utilities/sanity_check_matmul_filter'



kernel_size = (1, 3, 3)

filter = np.zeros((1, 1, kernel_size[1], kernel_size[2]), dtype=float)
filter[0, 0] = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]
                         ])
# kernel_size = (1, 5, 5)
#
# filter = np.zeros((1, 1, kernel_size[1], kernel_size[2]), dtype=float)
# filter[0, 0] = np.array([[0, 0, 0, 0, 0],
#                          [0, 1, 0, -1, 0],
#                          [0, 2, 0, -2, 0],
#                          [0, 1, 0, -1, 0],
#                          [0, 0, 0, 0, 0]])
filter = torch.from_numpy(filter)



filter2 = np.zeros((1, 1, kernel_size[1], kernel_size[2]), dtype=float)
filter2[0, 0] = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]
                         ])

filter2 = torch.from_numpy(filter2)




image_size = (1, 256, 256)
image = np.array(Image.open('/home/gabras/3tconv/utilities/doggo.jpg').convert('L'))
image = torch.from_numpy(image)
image = image.unsqueeze(0)
image = image.unsqueeze(0)

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
    

def params_x_something(mult_w_filter, mult_w_image):

    if mult_w_filter:
        something_size = kernel_size
        something = filter
    elif mult_w_image:
        something_size = image_size
        something = image
        save = True
    else:
        print('ERROR: something is not defined')
        something_size = None
        something = None
        save = False

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


    print('something', something.shape)
    print('thata', new_theta.shape)
    print('result', result.shape)

    if save:
        img = Image.fromarray(np.array(result[0][0].cpu(), dtype=np.uint8), mode='L')
        name = 'test.jpg'
        img.save(os.path.join(save_path, name))

    return result


# params_x_something(mult_w_filter=True, mult_w_image=False)  # works!
# params_x_something(mult_w_filter=False, mult_w_image=True)  # works!

def filter_x_image():
    the_image = image.cuda(device)
    the_filter = filter.cuda(device)
    the_filter2 = filter2.cuda(device)


    the_image = the_image.type(torch.float32)
    the_filter = the_filter.type(torch.float32)
    the_filter2 = the_filter2.type(torch.float32)

    result_conv2d = F.conv2d(the_image, the_filter, stride=1)
    result2_conv2d = F.conv2d(the_image, the_filter2, stride=1)

    result1 = np.array(result_conv2d[0][0].cpu())
    result2 = np.array(result2_conv2d[0][0].cpu())

    result = np.sqrt(np.square(result1)+np.square(result2))

    # img = Image.fromarray(np.array(result_conv2d[0][0].cpu()), mode='L')
    img = Image.fromarray(result, mode='L')
    # name = 'filter_x_image.jpg'
    name = 'sobel.jpg'
    img.save(os.path.join(save_path, name))



filter_x_image()