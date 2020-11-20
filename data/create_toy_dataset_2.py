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


MARGIN = 5
SIDE = 200

def generate_single_dot_w_hat(the_class, seed, side=200):
    '''
    generates a single dot with a hat at a random position
    applies random scale, translation and rotation
    '''
    np.random.seed(seed)

    if the_class == 'scale':
        lower = 20
        upper = 110
        # scale: 30 each side extra
    else:
        lower = 20
        upper = 170

    size_dot = np.random.randint(lower, upper)

    # loc_x = np.random.randint(upper, side-upper)
    # loc_y = np.random.randint(upper, side-upper)
    loc_x = (SIDE - size_dot) // 2
    loc_y = (SIDE - size_dot) // 2

    image = Image.new('L', (side, side))
    canvas = ImageDraw.Draw(image)
    canvas.ellipse((loc_x, loc_y, loc_x+size_dot, loc_y+size_dot), fill='white')
    triangle_radius = size_dot // 5
    # canvas.regular_polygon((loc_x, loc_y, size_dot//5), 3, -75, 'white')
    canvas.regular_polygon((loc_x+size_dot//2, loc_y, triangle_radius), 3, fill='white')


    frame_save_path = os.path.join(PP.dots_samples, 'dotwhat1.jpg')
    image.save(frame_save_path)

    # rotate
    angle = np.random.randint(0, 360)
    direction = np.random.randint(0, 2)
    if direction == 0:
        direction = -1
    image_rot = image.rotate(direction*angle, resample=Image.BILINEAR)

    frame_save_path = os.path.join(PP.dots_samples, 'dotwhat2.jpg')
    image_rot.save(frame_save_path)

    # scale
    scale_amount = np.random.randint(1, 6)

    if direction == -1:
        frame = image_rot.resize((side+scale_amount, side+scale_amount), Image.BILINEAR)
        frame = frame.crop((0+scale_amount, 0+scale_amount, side-scale_amount, side-scale_amount))
    else:
        frame = image_rot.crop((0+scale_amount, 0+scale_amount, side-scale_amount, side-scale_amount))

    frame = frame.resize((side, side), Image.BILINEAR)

    frame_save_path = os.path.join(PP.dots_samples, 'dotwhat3.jpg')
    frame.save(frame_save_path)

    bbox = list(frame.getbbox())

    dot_w_hat = frame.crop(bbox)
    dot_w, dot_h = dot_w_hat.size
    max_dim = max(dot_w, dot_h)
    square = Image.new('L', (max_dim, max_dim))
    offset = ((max_dim - dot_w) // 2, (max_dim - dot_h) // 2)
    square.paste(dot_w_hat, offset)

    # if max_dim == dot_w:
    #     bbox[1] = bbox[1] - offset[1]
    #     bbox[3] = bbox[3] + offset[1]
    # elif max_dim == dot_h:
    #     bbox[0] = bbox[0] - offset[0]
    #     bbox[2] = bbox[2] + offset[0]

    # save sample
    # frame_save_path = os.path.join(PP.dots_samples, 'dot_hat_%d_%d_%d_%d.jpg' % (loc_x, loc_y, angle, scale_amount))
    frame_save_path = os.path.join(PP.dots_samples, 'dot_hat_square.jpg')
    square.save(frame_save_path)

    # bbox = tuple(bbox)

    return square

# generate_single_dot_w_hat(8)

def generate_sequence(the_class, num_frames, seed, window_side=200, cropped=32):
    '''
    samples random parameters
    applies transformations to base
    saves sequence
    '''

    np.random.seed(seed)
    dot_w_hat = generate_single_dot_w_hat(the_class, seed)
    margin = 5
    dot_w, dot_h = dot_w_hat.size
    bbox = [0, 0, 0, 0]

    annotation, direction, direction_ver, direction_hor, speed = None, None, 0, 0, None

    if the_class == 'translate':

        # sample starting location
        bbox[0] = np.random.randint(MARGIN, window_side-dot_w-MARGIN)
        bbox[1] = np.random.randint(MARGIN, window_side-dot_h-MARGIN)
        bbox[2] = bbox[0] + dot_w
        bbox[3] = bbox[1] + dot_h

        direction_hor = np.random.randint(-1, 2)
        if direction_hor == 0:
            while direction_ver == 0:
                direction_ver = np.random.randint(-1, 2)
        else:
            direction_ver = 0

        speed = np.random.randint(1, 4)
        annotation = [seed, direction_hor, direction_ver, speed]

    elif the_class == 'rotate':
        bbox[0] = np.random.randint(MARGIN, window_side-dot_w-MARGIN)
        bbox[1] = np.random.randint(MARGIN, window_side-dot_h-MARGIN)
        bbox[2] = bbox[0] + dot_w
        bbox[3] = bbox[1] + dot_h

        direction = np.random.randint(0, 2)
        if direction == 0:
            direction = -1
        speed = np.random.randint(1, 4)
        # print('%s: direction: %d, speed: %d' % (the_class, direction, speed))
        annotation = [seed, direction, direction, speed]

    elif the_class == 'scale':

        if num_frames*2 + dot_w > window_side - margin*2:
            # too big
            direction = -1
            pass
        elif dot_w - num_frames*2 < margin*2:
            # too small
            direction = 1
        else:
            direction = np.random.randint(0, 2)
            if direction == 0:
                direction = -1


        if direction == 1:
            bbox[0] = np.random.randint(MARGIN+num_frames, window_side-dot_w-MARGIN-num_frames)
            bbox[1] = np.random.randint(MARGIN+num_frames, window_side-dot_h-MARGIN-num_frames)
            bbox[2] = bbox[0] + dot_w
            bbox[3] = bbox[1] + dot_h
        else:
            bbox[0] = np.random.randint(MARGIN, window_side-dot_w-MARGIN)
            bbox[1] = np.random.randint(MARGIN, window_side-dot_h-MARGIN)
            bbox[2] = bbox[0] + dot_w
            bbox[3] = bbox[1] + dot_h

        speed = 1
        # speed = np.random.randint(1, 3)
        # print('%s: direction: %d, speed: %d' % (the_class, direction, speed))
        annotation = [seed, direction, direction, speed]

    else:
        print('error: class not recognized')

    all_frames_in_sequence = np.zeros((num_frames, cropped, cropped), dtype=np.uint8)
    black_image = Image.new('L', (window_side, window_side))


    if the_class == 'translate':
        previous_location = bbox
        for f in range(num_frames):
            base_image = black_image
            left = speed * direction_hor + previous_location[0]
            up = speed * direction_ver + previous_location[1]
            right = speed * direction_hor + previous_location[2]
            down = speed * direction_ver + previous_location[3]

            if left < margin or right > window_side+margin:
                direction_hor = direction_hor * -1
                left = speed * direction_hor + previous_location[0]
                right = speed * direction_hor + previous_location[2]

            if up < margin or down > window_side+margin:
                direction_ver = direction_ver * -1
                up = speed * direction_ver + previous_location[1]
                down = speed * direction_ver + previous_location[3]

            previous_location = (left, up, right, down)
            base_image.paste(dot_w_hat, (previous_location[0], previous_location[1]))
            frame = base_image.resize((cropped, cropped), Image.BILINEAR)
            all_frames_in_sequence[f] = np.array(frame)
            
    elif the_class == 'rotate':

        for f in range(num_frames):
            base_image = black_image

            dot_w_hat_rot = dot_w_hat.rotate(f*direction*speed, resample=Image.BILINEAR)
            base_image.paste(dot_w_hat_rot, (bbox[0], bbox[1]))
            frame = base_image.resize((cropped, cropped), Image.BILINEAR)
            all_frames_in_sequence[f] = np.array(frame)

    elif the_class == 'scale':

        for f in range(num_frames):
            base_image = black_image
            dot_w_hat_sc = dot_w_hat.resize((dot_w + direction*f, dot_h + direction*f), Image.BILINEAR)
            offset = ((bbox[0]-direction*f), (bbox[1]-direction*f))

            base_image.paste(dot_w_hat_sc, offset)
            frame = base_image.resize((cropped, cropped), Image.BILINEAR)
            all_frames_in_sequence[f] = np.array(frame)

    return all_frames_in_sequence, annotation


def make_dataset(which, num_samples_per_class, num_frames, crop_side):

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

        np.random.seed(class_seeds[i])
        video_seeds = np.random.randint(0, 1000000, num_samples_per_class)

        avi_class_path = os.path.join(avi_which_path, c)
        frames_class_path = os.path.join(frames_which_path, c)
        U.opt_makedirs(avi_class_path)
        U.opt_makedirs(frames_class_path)

        # for s in tqdm(range(num_samples_per_class)):
        for s in range(num_samples_per_class):
            all_frames_array, annotation = generate_sequence(c, num_frames, video_seeds[s], 200, crop_side)
            # all_frames_array, annotation = generate_sequence_dots(c, num_frames, video_seeds[s], side, parameters=params)
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


which = 'train'
num_samples_per_class = 3
num_frames = 30
crop_side = 33
make_dataset(which, num_samples_per_class, num_frames, crop_side)