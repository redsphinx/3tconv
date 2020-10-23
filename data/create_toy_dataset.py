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


def sample_base_image(seed, height, width, randomize):
    if randomize:
        np.random.seed(seed)
        rotation = np.random.randint(0, 360)
        frequency = np.random.randint(5, 80) / 100  # TODO: how to deal with going out of bounds?
        phase = np.random.randint(0, 10) / 10 * np.pi
    else:
        # rotation = 225
        rotation = 45
        frequency = 0.5
        phase = np.pi

    radius = (int(width / 2.0), int(height / 2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0] + 1), range(-radius[1], radius[1] + 1))
    # [x, y] = rotate_grid([x, y], rotation)

    return [x, y], frequency, rotation, phase


def fix_shape(np_arr, desired_shape):
    if np_arr.shape != desired_shape:
        np_arr = np_arr[0:desired_shape[0], 0:desired_shape[1]]

    return np_arr



def generate_single_sequence(the_class, height, width, frames, seed, randomize=True):
    amplitude = 1
    # make base image -> sample phase, frequency and rotation
    # apply the sin wave later
    [x, y], frequency, rotation, phase = sample_base_image(seed, height, width, randomize)

    # x = fix_shape(x, (height, width))
    # y = fix_shape(y, (height, width))

    transformation_sequence = np.zeros((frames-1, 2, 3))
    # transformation_sequence = np.zeros((frames-1, 3, 2))

    if the_class == 'translate':
        horizontal = 0
        vertical = 0
        if randomize:
            while horizontal == 0 and vertical == 0:
                horizontal = np.random.randint(-1, 2) # -1 = right, 1 = left
                vertical = np.random.randint(-1, 2)

            speed_per_frame = np.random.randint(1, 4) # move between 1 and 3 pixels
            rotation = np.random.randint(0, 360)
        else:
            horizontal = 1
            vertical = 0  # 1 = up, -1 = down
            speed_per_frame = 1
            rotation = 0

        rot_rad = rotation * np.pi / 180

        for i in range(1, frames):
            transformation_sequence[i-1] = np.array([[1, 0, i * horizontal * speed_per_frame],
                                                     [0, 1, i * vertical * speed_per_frame]])

            # transformation_sequence[i-1] = np.array([[np.cos(rot_rad),  np.sin(rot_rad), i * horizontal * speed_per_frame],
            #                                          [-np.sin(rot_rad), np.cos(rot_rad), i * vertical * speed_per_frame]])

        info = 'horizontal: %d, vertical: %d, speed: %d' % (horizontal, vertical, speed_per_frame)

    elif the_class == 'rotate':
        def rot_rad(ind=1):
            speed_in_rad = ind * speed_per_frame * np.pi / 180
            rad = direction * speed_in_rad
            return rad

        if randomize:
            direction = np.random.randint(0, 2)
            if direction == 0:  # 1 clockwise
                direction = -1  # -1 counterclockwise
            speed_per_frame = np.random.randint(1, 4) # move between 1 and 3 degrees
            center_x = 0 # np.random.randint(11, 22)
            center_y = 0 # np.random.randint(11, 22)
        else:
            direction = -1
            speed_per_frame = 1
            center_x = 0 # width // 2
            center_y = 0 # height // 2


        # delta_trafo = np.array(
        #     [[np.cos(rot_rad()), np.sin(rot_rad()), center_x * np.cos(rot_rad()) - center_y * np.sin(rot_rad())],
        #      [-np.sin(rot_rad()), np.cos(rot_rad()), center_x * np.sin(rot_rad()) + center_y * np.cos(rot_rad())]])

        for i in range(1, frames):
            transformation_sequence[i - 1] = np.array([[np.cos(rot_rad(i)), np.sin(rot_rad(i)), center_x * np.cos(rot_rad(i)) - center_y * np.sin(rot_rad(i))],
                                                       [-np.sin(rot_rad(i)), np.cos(rot_rad(i)), center_x * np.sin(rot_rad(i)) + center_y * np.cos(rot_rad(i))]])

            # apply translation to base mesh for off-center rotation
        translation_trafo = np.array([[1, 0, center_x * speed_per_frame],
                                      [0, 1, center_y * speed_per_frame]])

        [x, y] = np.einsum('ji, mni -> jmn', translation_trafo, np.dstack([x, y, np.ones(x.shape)]))

        info = 'direction: %d, center: (%d, %d), speed: %d' % (direction, center_x, center_y, speed_per_frame)


    elif the_class == 'scale':
        center_x = 0 # np.random.randint(11, 22)
        center_y = 0 # np.random.randint(11, 22)

        if randomize:
            direction = np.random.randint(0, 2)
            if direction == 0:  # 1 zoom out
                direction = -1  # -1 zoom in
            speed_per_frame = np.random.randint(1, 4)/100 # between 1 and 3 ???
        else:
            direction = 1
            speed_per_frame = 0.03


        # delta_trafo = np.array([[scaling, scaling, center_x*scaling-center_y*scaling],
        #                        [scaling, scaling, center_x*scaling+center_y*scaling]])

        for i in range(1, frames):
            # transformation_sequence[i-1] = np.array([[np.power(scaling, i), np.power(scaling, i), center_x*np.power(scaling, i)-center_y*np.power(scaling, i)],
            #                                         [np.power(scaling, i), np.power(scaling, i), center_x*np.power(scaling, i)+center_y*np.power(scaling, i)]])
            transformation_sequence[i-1] = np.array([[1 + direction * i * speed_per_frame, 0, 0],
                                                     [0, 1 + direction * i * speed_per_frame, 0]])
            # transformation_sequence[i-1] = np.array([[1 + direction * np.power(speed_per_frame, i), 0, 0],
            #                                          [0, 1 + direction * np.power(speed_per_frame, i), 0]])


        # apply translation to base mesh for off-center scaling
        # translation_trafo = np.array([[1, 0, center_x * speed_per_frame],
        #                               [0, 1, center_y * speed_per_frame]])
        #
        # [x, y] = np.einsum('ji, mni -> jmn', translation_trafo, np.dstack([x, y, np.ones(x.shape)]))

        info = 'direction: %d, center: (%d, %d), speed: %f' % (direction, center_x, center_y, speed_per_frame)

    # transformation_sequence = create_transformation_sequence(the_class, frames, delta_trafo)

    x_h = x.shape[0]
    x_w = x.shape[1]

    all_frames_in_sequence = np.zeros((frames, x_h, x_w), dtype=np.uint8)
    # add the base frame
    frame = amplitude * np.cos(frequency * x + frequency * y + phase)
    all_frames_in_sequence[0] = frame

    for f in range(frames-1):
        # apply delta transformation
        transformation = np.eye(3)
        transformation[0:2] = transformation_sequence[f]

        # mesh = np.einsum('ji, mni -> jmn', transformation_sequence[f], np.dstack([x, y]))
        # mesh = np.einsum('ij, mni -> imn', transformation_sequence[f], np.dstack([x, y]))
        # mesh = np.einsum('ji, mni -> jmn', transformation_sequence[f], np.dstack([x, y, np.ones(x.shape)]))
        mesh = np.einsum('ji, mni -> jmn', transformation, np.dstack([x, y, np.ones(x.shape)]))
        # fill mesh with sin wave
        frame = amplitude * np.cos(frequency * mesh[0] + frequency * mesh[1] + phase)

        # testing the rotation after




        frame = U.normalize_between(frame, frame.min(), frame.max(), 0, 255)
        # save as jpg
        frame = np.asarray(frame, dtype=np.uint8)
        # im = Image.fromarray(frame, mode='L')
        # savpath = os.path.join(PP.gaff_samples, 'fr_%d.jpg' % f)
        # im.save(savpath)

        all_frames_in_sequence[f] = frame

    # save as avi
    # save_path = os.path.join(PP.gaff_samples, "%s_%s_%s_%s.avi" % (the_class, str(rotation), str(phase), str(frequency)))
    save_path = os.path.join(PP.gaff_samples, "%s_%d.avi" % (the_class, np.random.randint(1000, 9999)))
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
# make_sample(amplitude=amplitude, phase=phase, frequency=frequency, rotation=10)


the_class = 'translate'
# the_class = 'rotate'
# the_class = 'scale'
height, width = 32, 32
frames = 30
seed = 420
generate_single_sequence(the_class, height, width, frames, seed, randomize=False)

'''
SEEDS
train   1
val     2
test    3
Given each seed, each split generates N number of new seeds equal to the number of samples in the split.
Each sequence in the split uses said seed to generate the specific parameters. 

'''
