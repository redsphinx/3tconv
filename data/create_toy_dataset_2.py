# This is a toy dataset with 3 classes so that we can validate 3TConv
# This is the single dot version

from tqdm import tqdm
import numpy as np
import os
import subprocess
from PIL import Image, ImageDraw, ImageFilter
import tkinter as tk
import time
from datetime import datetime
import skvideo.io as skvid
import cv2 as cv
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import config.paths as PP
from config.base_config import ProjectVariable
from running import main_file
import utilities.utils as U


def generate_single_dot_w_hat(seed, side=200):
    '''
    generates a single dot with a hat at a random position
    applies random scale, translation and rotation
    '''
    np.random.seed(seed)

    lower = 20
    upper = 60

    size_dot = np.random.randint(lower, upper)
    loc_x = np.random.randint(upper, side-upper)
    loc_y = np.random.randint(upper, side-upper)

    image = Image.new('L', (side, side))
    canvas = ImageDraw.Draw(image)
    canvas.ellipse((loc_x, loc_y, loc_x+size_dot, loc_y+size_dot), fill='white')
    canvas.regular_polygon((loc_x, loc_y, size_dot//2), 3, -75, 'white')

    # rotate
    angle = np.random.randint(0, 361)
    direction = np.random.randint(0, 2)
    if direction == 0:
        direction = -1
    image_rot = image.rotate(direction*angle, resample=Image.BILINEAR)

    # scale
    scale_amount = np.random.randint(1, 6)

    if direction == -1:
        frame = image_rot.resize((side+scale_amount, side+scale_amount), Image.BILINEAR)
        frame = frame.crop((0+scale_amount, 0+scale_amount, side-scale_amount, side-scale_amount))
    else:
        frame = image_rot.crop((0+scale_amount, 0+scale_amount, side-scale_amount, side-scale_amount))
        frame = frame.resize((side, side), Image.BILINEAR)

    bbox = frame.getbbox()

    # save sample
    # frame_save_path = os.path.join(PP.dots_samples, 'dot_hat_%d_%d_%d_%d.jpg' % (loc_x, loc_y, angle, scale_amount))
    # frame.save(frame_save_path)

    return frame, bbox


def generate_sequence(the_class, num_frames, seed, window_side=200, cropped=32):
    '''
    samples random parameters
    applies transformations to base
    saves sequence
    '''

    np.random.seed(seed)

    annotation, direction, direction_ver, direction_hor, speed = None, None, 0, 0, None

    if the_class == 'translate':

        direction_hor = np.random.randint(-1, 2)
        if direction_hor == 0:
            while direction_ver == 0:
                direction_ver = np.random.randint(-1, 2)
        else:
            direction_ver = 0

        speed = np.random.randint(1, 4)
        annotation = [seed, direction_hor, direction_ver, speed]

    elif the_class == 'rotate':
        direction = np.random.randint(0, 2)
        if direction == 0:
            direction = -1
        speed = np.random.randint(1, 4)
        # print('%s: direction: %d, speed: %d' % (the_class, direction, speed))
        annotation = [seed, direction, direction, speed]

    elif the_class == 'scale':
        direction = np.random.randint(0, 2)
        if direction == 0:
            direction = -1
        speed = np.random.randint(1, 3)
        # print('%s: direction: %d, speed: %d' % (the_class, direction, speed))
        annotation = [seed, direction, direction, speed]

    else:
        print('error: class not recognized')


    dot_w_hat, bbox = generate_single_dot_w_hat(seed)
    

    '''
    for each transformation
        if scale
        also determine the speed and direction of scaling depending on the initial size of the dot
        depending on the lower and upper bound of the dot sizes        
        if translate, if it hits a boundary, change the direction
        make sure that the white does not overlap with a boundary
        crop to windowsize
    '''

    all_frames_in_sequence = np.zeros((num_frames, cropped, cropped), dtype=np.uint8)




    pass


def make_dataset(which, num_samples_per_class, num_frames, side, parameters=None):

    print('which: %s' % which)

    which_seeds = [6, 42, 420]
    if which == 'train':
        master_seed = which_seeds[0]

    elif which == 'val':
        master_seed = which_seeds[1]

    else:
        master_seed = which_seeds[2]


    np.random.seed(master_seed)

    avi_which_path = os.path.join(PP.dots_dataset_avi, which)
    frames_which_path = os.path.join(PP.dots_dataset_frames, which)

    classes = ['rotate', 'scale', 'translate']
    class_seeds = np.random.randint(0, 10000, 3)

    for i, c in enumerate(classes):
        print('class: %s' % c)

        if parameters is not None:
            if c == 'translate':
                params = parameters[0:3]
            elif c == 'rotate':
                params = parameters[3:6]
            elif c == 'scale':
                params = parameters[6:]
            else:
                print('this is a weird class %s' % c)
                params = None

        else:
            params = None


        np.random.seed(class_seeds[i])
        video_seeds = np.random.randint(0, 1000000, num_samples_per_class)

        avi_class_path = os.path.join(avi_which_path, c)
        frames_class_path = os.path.join(frames_which_path, c)
        U.opt_makedirs(avi_class_path)
        U.opt_makedirs(frames_class_path)

        # for s in tqdm(range(num_samples_per_class)):
        for s in range(num_samples_per_class):
            all_frames_array, annotation = generate_sequence_dots(c, num_frames, video_seeds[s], side, parameters=params)
            choose_frame = np.random.randint(0, num_frames)
            frame = all_frames_array[choose_frame]

            # save avi
            if c == 'rotate':
                num = s
            elif c == 'scale':
                num = s + num_samples_per_class
            else:
                num = s + 2*num_samples_per_class

            vid = '%05d' % (num + 1)
            avi_save_path = os.path.join(avi_class_path, '%s.avi' % vid)
            skvid.vwrite(avi_save_path, all_frames_array)

            # save avi annotation
            avi_annotation = os.path.join(PP.dots_dataset_avi, 'annotations_%s.txt' % which)
            with open(avi_annotation, 'a') as my_file:
                # vid,class,seed,direction_hor,direction_ver,speed,frames (if translate)
                # vid,class,seed,direction,direction,speed,frames (else)
                line = '%s,%s,%d,%d,%d,%d,%d\n' % (vid, c, annotation[0], annotation[1], annotation[2], annotation[3], num_frames)
                my_file.write(line)

            # save frame
            frame_save_path = os.path.join(frames_class_path, '%s.jpg' % vid)
            frame = Image.fromarray(frame, mode='L')
            frame.save(frame_save_path)

            # save frame annotation
            frame_annotation = os.path.join(PP.dots_dataset_frames, 'annotations_%s.txt' % which)
            with open(frame_annotation, 'a') as my_file:
                # vid,class,frame
                line = '%s,%s,%d\n' % (vid, c, choose_frame)
                my_file.write(line)

