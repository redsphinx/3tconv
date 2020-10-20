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

    # Clockwise 2D rotation matrix
    # row x column
    RotMatrix = np.array([[np.cos(rot_rad),  np.sin(rot_rad)],
                          [-np.sin(rot_rad), np.cos(rot_rad)]])

    return np.einsum('ji, mni -> jmn', RotMatrix, np.dstack(grid))


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


def create_transformation_sequence(frames, delta_trafo):

    transformation_sequence = np.zeros((frames, 2, 3))

    for i in range(1, frames):
        transformation_sequence[i] = i * delta_trafo

    return transformation_sequence


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
    # amplitude = 1
    # base_mesh = amplitude * np.cos(frequency * x + frequency * y + phase)

    return [x, y], frequency, rotation, phase


def generate_single_sequence(which, the_class, num_samples, height, width, frames, seed):
    amplitude = 1
    # make base image -> sample phase, frequency and rotation
    # apply the sin wave later
    [x, y], frequency, rotation, phase = sample_base_image(seed, height, width)

    if the_class == 'translate':
        horizontal = 0
        vertical = 0
        while horizontal == 0 and vertical == 0:
            horizontal = np.random.randint(-1, 2)
            vertical = np.random.randint(-1, 2)

        speed_per_frame = np.random.randint(1, 4) # move between 1 and 3 pixels

        delta_trafo = np.array([[1, 0, horizontal * speed_per_frame],
                                [0, 1, vertical * speed_per_frame]])

        info = 'horizontal: %d, vertical: %d, speed: %d' % (horizontal, vertical, speed_per_frame)

    elif the_class == 'rotate':
        direction = np.random.randint(0, 2)
        if direction == 0:
            direction = -1
        speed_per_frame = np.random.randint(1, 4) # move between 1 and 3 degrees
        speed_in_rad = speed_per_frame * np.pi / 180
        rot_rad = direction * speed_in_rad

        center_x = np.random.randint(11, 22)
        center_y = np.random.randint(11, 22)

        delta_trafo = np.array([np.cos(rot_rad), np.sin(rot_rad), center_x*np.cos(rot_rad)-center_y*np.sin(rot_rad)],
                               [-np.sin(rot_rad), np.cos(rot_rad), center_x*np.sin(rot_rad)+center_y*np.cos(rot_rad)])

        # apply translation to base mesh for off-center rotation
        translation_trafo = np.array([[1, 0, center_x * speed_per_frame],
                                      [0, 1, center_y * speed_per_frame]])

        [x, y] = np.einsum('ji, mni -> jmn', translation_trafo, np.dstack([x, y]))

        info = 'direction: %d, center: (%d, %d), speed: %d' % (direction, center_x, center_y, speed_per_frame)


    elif the_class == 'scale':


        center_x = np.random.randint(11, 22)
        center_y = np.random.randint(11, 22)
        direction = np.random.randint(0, 2)
        if direction == 0:
            direction = -1
        speed_per_frame = np.random.randint(1, 4)/10 # between 1 and 3 ???

        scaling = 1 + direction*speed_per_frame

        delta_trafo = np.array([scaling, scaling, center_x*scaling-center_y*scaling],
                               [scaling, scaling, center_x*scaling+center_y*scaling])

        # apply translation to base mesh for off-center scaling
        translation_trafo = np.array([[1, 0, center_x * speed_per_frame],
                                      [0, 1, center_y * speed_per_frame]])

        [x, y] = np.einsum('ji, mni -> jmn', translation_trafo, np.dstack([x, y]))

        info = 'direction: %d, center: (%d, %d), speed: %d' % (direction, center_x, center_y, speed_per_frame)

    else:
        delta_trafo = None

    transformation_sequence = create_transformation_sequence(frames, delta_trafo)

    # TODO: channels
    all_frames_in_sequence = np.zeros((frames, height, width), dtype=np.uint8)
    # add the base frame
    frame = amplitude * np.cos(frequency * x + frequency * y + phase)
    all_frames_in_sequence[0] = frame

    for f in range(frames-1):
        # apply delta transformation
        mesh = np.einsum('ji, mni -> jmn', transformation_sequence[f], np.dstack([x, y]))
        # fill mesh with sin wave
        frame = amplitude * np.cos(frequency * mesh[0] + frequency * mesh[1] + phase)
        all_frames_in_sequence[f] = frame

    # save
    save_path = os.path.join(PP.gaff_samples, "%s_%s_%s_%s.avi" % (the_class, str(rotation), str(phase), str(frequency)))
    skvid.vwrite(save_path, all_frames_in_sequence)
    print('class: %s\n'
          'base rotation, phase, frquency: %d, %f, %f\n'
          'transformation: %s' % (the_class, rotation, phase, frequency, info))
    # TODO: save metadata in file




# # pi = np.pi
# amplitude = 1 # idk what this does
# phase = 1.1 # controls horizontal movement
# frequency = 0.52 # controls stripe density
# rotation = 0
#
# # theta = pi / 4
# # frequency = [np.cos(theta), np.sin(theta)]
# # frequency = np.sin(theta)
# # frequency = theta
# # make_sample(amplitude=1, phase=pi/2, frequency=frequency)
#
# make_sample(amplitude=amplitude, phase=phase, frequency=frequency, rotation=None)



'''
SEEDS
train   1
val     2
test    3
Given each seed, each split generates N number of new seeds equal to the number of samples in the split.
Each sequence in the split uses said seed to generate the specific parameters. 

'''
