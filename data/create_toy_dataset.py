# This is a toy dataset with 3 classes so that we can validate 3TConv
import numpy as np
import os
from PIL import Image
import time
from datetime import datetime
import skvideo.io as skvid
import cv2 as cv
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import config.paths as PP
import utilities.utils as U

'''
3 classes

sinusoid on 2D plane, moving through time

class 1: translation
class 2: rotation
class 3: scaling

for each class there are variations
- band density (frequency)
- orientation
- initial scale (how thick/thin bands are)
- speed at which the transformation takes place
- direction in which the transformation takes place




class 1: 4 directions, left right up down
class 2: clockwise, counterclockwise, point of rotation
class 3: scale up or down, origin of scaling


'''

def generate_sin_wave(amplitude, phase, frequency, rotate=None, height=32, width=32):
    radius = (int(width / 2.0), int(height / 2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0] + 1), range(-radius[1], radius[1] + 1))
    if rotate is not None:
        [x, y] = rotate_grid([x, y], rotate)

    # result = amplitude * np.cos(frequency[0] * x  + frequency[1] * y + phase)
    result = amplitude * np.cos(frequency * x + frequency * y + phase)

    result = U.normalize_between(result, result.min(), result.max(), 0, 255)

    return result


def rotate_grid(grid, degrees):
    # degree to radians
    rot_rad = degrees * np.pi / 180

    # Clockwise, 2D rotation matrix
    RotMatrix = np.array([[np.cos(rot_rad),  np.sin(rot_rad)],
                          [-np.sin(rot_rad), np.cos(rot_rad)]])

    return np.einsum('ji, mni -> jmn', RotMatrix, np.dstack(grid))


def zooming(scale):
    pass





def make_sample(amplitude, phase, frequency, rotation=None):
    image = generate_sin_wave(amplitude, phase * np.pi, frequency * np.pi, rotation)

    plt.figure()
    plt.axis('off')

    plt.imshow(image, cmap=plt.gray(), interpolation='quadric')

    # rotation


    save_path = os.path.join(PP.gaff_samples, "im_%s_%s_%s.jpg" % (str(amplitude), str(phase), str(frequency)))
    plt.savefig(save_path)

    # im = Image.fromarray(image, mode='L')
    # im.save(save_path)


def sample_base_image(seed, height, width):
    # phase, frequency, rotation

    np.random.seed(seed)
    rotation = np.random.randint(0, 360)
    frequency = np.random.randint(5, 80) / 100  # TODO: how to deal with going out of bounds?
    phase = np.random.randint(0, 10) / 10 * np.pi


    radius = (int(width / 2.0), int(height / 2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0] + 1), range(-radius[1], radius[1] + 1))
    [x, y] = rotate_grid([x, y], rotation)

    # result = amplitude * np.cos(frequency[0] * x  + frequency[1] * y + phase)
    amplitude = 1
    base_mesh = amplitude * np.cos(frequency * x + frequency * y + phase)

    return base_mesh


def generate_dataset(which, the_class, num_samples, height, width, frames, seed):

    # make base image -> sample phase, frequency and rotation
    base_mesh = sample_base_image(seed, height, width)

    # if class == translate
    #     sample horizontal and vertical direction, speed
    #     calculate deltas
    if the_class == 'translate':
        pass

    # if class == rotate
    #     sample rotation center, direction, speed
    #     calculate deltas
    elif the_class == 'rotate':
        pass

    #  if class == scale
    #     sample scale center, direction, speed
    #     calculate deltas
    elif the_class == 'scale':
        pass

    # apply transformations only on base image
    # save sequence, transformations, seed
    pass



# pi = np.pi
amplitude = 1 # idk what this does
phase = 3.14 *2 # controls horizontal movement
frequency = 0.05 # controls stripe density
rotation = 0

# theta = pi / 4
# frequency = [np.cos(theta), np.sin(theta)]
# frequency = np.sin(theta)
# frequency = theta
# make_sample(amplitude=1, phase=pi/2, frequency=frequency)

make_sample(amplitude=amplitude, phase=phase, frequency=frequency, rotation=None)



'''
SEEDS
train   1
val     2
test    3
Given each seed, each split generates N number of new seeds equal to the number of samples in the split.
Each sequence in the split uses said seed to generate the specific parameters. 

'''
