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


IM_HEIGHT = 32
IM_WIDTH = 32
NUM_FRAMES = 30




def generate_sin_wave(amplitude, phase, frequency):
    radius = (int(IM_WIDTH/2.0), int(IM_HEIGHT/2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0]+1), range(-radius[1], radius[1]+1))
    # result = amplitude * np.cos(frequency[0] * x  + frequency[1] * y + phase)
    result = amplitude * np.cos(frequency * x  + frequency * y + phase)

    # result = U.normalize_between(result, result.min(), result.max(), 0, 255)

    return result


def make_sample(amplitude, phase, frequency):
    image = generate_sin_wave(amplitude, phase*np.pi, frequency*np.pi)

    plt.figure()
    plt.axis('off')

    plt.imshow(image, cmap=plt.gray(), interpolation="bicubic")
    save_path = os.path.join(PP.gaff_samples, "im_%s_%s_%s.jpg" % (str(amplitude), str(phase), str(frequency)))
    plt.savefig(save_path)


    # im = Image.fromarray(image, mode='L')
    # im.save(save_path)

# pi = np.pi
amplitude = 1 # idk what this does
phase = 0.25 # controls horizontal movement
frequency = 1 # controls stripe density

# theta = pi / 4
# frequency = [np.cos(theta), np.sin(theta)]
# frequency = np.sin(theta)
# frequency = theta
# make_sample(amplitude=1, phase=pi/2, frequency=frequency)

make_sample(amplitude=amplitude, phase=phase, frequency=frequency)

