# This is a toy dataset with 3 classes so that we can validate 3TConv
from tqdm import tqdm
import numpy as np
import os
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
import utilities.utils as U


def generate_dots(the_class, direction=None, speed=None, num_frames=30, window_side=32, really_big=False, big_factor=5):

    '''
    depending on the class, generate an image and then slide over it with a window
    total canvas size
    translate: maxspeed * frames * 2
    rotate: sqrt(23^2+32^2)
    scale:

    '''
    if the_class == 'translate':
        if speed is None:
            speed = 1

        full_side = window_side + speed * num_frames * 2
        radius_dot_min = np.random.randint(1, 6)
        radius_dot_max = radius_dot_min + 1
        # the smaller the dots, the faster the speed, the more dots you need
        num_dots_low = speed * 1 / radius_dot_min * 60
        num_dots_high = speed * 1 / radius_dot_min * 70

        if really_big:
            full_side = full_side * big_factor
            radius_dot_min = radius_dot_min * big_factor
            radius_dot_max = radius_dot_max * big_factor
            num_dots_low = num_dots_low * big_factor**2
            num_dots_high = num_dots_high * big_factor**2

        num_dots = np.random.randint(num_dots_low, num_dots_high)

    elif the_class == 'rotate':
        full_side = int(np.sqrt(2 * window_side**2)) + 1
        num_dots = np.random.randint(5, 11)
        radius_dot_min = np.random.randint(1, 4)
        radius_dot_max = radius_dot_min + 1

        if really_big:
            full_side = full_side * big_factor
            num_dots = num_dots * big_factor**2
            radius_dot_min = radius_dot_min * big_factor
            radius_dot_max = radius_dot_max * big_factor

    elif the_class == 'scale':
        if speed is None:
            speed = 1
        full_side = window_side + speed * num_frames * 2

        num_dots = np.random.randint(10, 13)
        if direction == 1:
            radius_dot_min = 6
            radius_dot_max = 9
        else:
            radius_dot_min = 7
            radius_dot_max = 10

        if really_big:
            full_side = full_side * big_factor
            # num_dots = num_dots * big_factor**2
            num_dots = num_dots * big_factor*(big_factor-2)
            radius_dot_min = radius_dot_min * big_factor
            radius_dot_max = radius_dot_max * big_factor

    else:
        print('error: class not recognized')
        return None

    # canvas
    image = Image.new('L', (full_side, full_side))
    canvas = ImageDraw.Draw(image)

    sample_x = np.random.randint(5, full_side-6, num_dots)
    sample_y = np.random.randint(5, full_side-6, num_dots)
    sizes_dots = np.random.randint(radius_dot_min, radius_dot_max, num_dots)

    # draw on canvas
    for n in range(num_dots):
        canvas.ellipse((sample_x[n], sample_y[n], sample_x[n]+sizes_dots[n], sample_y[n]+sizes_dots[n]), fill='white')

    # tmp_path = os.path.join(PP.dots_samples, "%s_dots_%d.jpg" % (the_class, np.random.randint(1000, 9999)))
    # image.save(tmp_path)

    return image, canvas, full_side


def apply_distortion(image, full_side):
    angle = np.random.randint(10, 30)
    direction = np.random.randint(0, 2)
    if direction == 0:
        direction = -1

    retry = True
    count = 0
    image_rot = image.rotate(direction*angle, resample=Image.BILINEAR)

    while retry:
        frame = image_rot.crop((0+count, 0+count, full_side-count, full_side-count))

        if frame.size != (0, 0):
            retry = False
        else:
            count = count + 1

    frame = frame.resize((full_side, full_side), Image.BILINEAR)

    # add gaussian blur
    # if np.random.randint(0, 2):
    #     frame = frame.filter(ImageFilter.GaussianBlur(radius=1))

    return frame


def generate_sequence_dots(the_class, num_frames, seed, window_side, really_big=True, big_factor=5):

    np.random.seed(seed)

    big_window_side = window_side*big_factor
    annotation, direction, direction_ver, direction_hor, speed = None, None, None, None, None

    if the_class == 'translate':
        direction_hor = np.random.randint(-1, 2)
        direction_ver = np.random.randint(-1, 2)

        while direction_hor == 0 and direction_ver == 0:
            direction_hor = np.random.randint(-1, 2)
            direction_ver = np.random.randint(-1, 2)

        speed = np.random.randint(1, 4)
        direction = None
        # print('%s: horizontal: %d, vertical: %d, speed: %d' % (the_class, direction_hor, direction_ver, speed))
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

    image, canvas, full_side = generate_dots(the_class, direction, speed, num_frames, really_big=really_big)
    image = apply_distortion(image, full_side)
    all_frames_in_sequence = np.zeros((num_frames, window_side, window_side), dtype=np.uint8)

    if the_class == 'translate':
        
        if really_big:
            left_x = full_side // 2 - big_window_side // 2
            up_y = full_side // 2 - big_window_side // 2

            for f in range(num_frames):
                bbox_left = f * speed * direction_hor + left_x
                bbox_right = bbox_left + big_window_side
                bbox_up = f * speed * direction_ver + up_y
                bbox_down = bbox_up + big_window_side

                frame = image.crop((bbox_left, bbox_up, bbox_right, bbox_down))
                # resize
                frame = frame.resize((window_side, window_side), Image.BILINEAR)
                # for avi
                all_frames_in_sequence[f] = np.array(frame)
            
        else:
            left_x = full_side // 2 - window_side // 2
            up_y = full_side // 2 - window_side // 2
    
            for f in range(num_frames):
                bbox_left = f * speed * direction_hor + left_x
                bbox_right = bbox_left + window_side
                bbox_up = f * speed * direction_ver + up_y
                bbox_down = bbox_up + window_side
    
                frame = image.crop((bbox_left, bbox_up, bbox_right, bbox_down))
    
                # for avi
                all_frames_in_sequence[f] = np.array(frame)

    elif the_class == 'rotate':
        if really_big:
            bbox_left = full_side // 2 - big_window_side // 2
            bbox_up = full_side // 2 - big_window_side // 2
            bbox_right = bbox_left + big_window_side
            bbox_down = bbox_up + big_window_side

            for f in range(num_frames):
                image_rot = image.rotate(f * speed * direction, resample=Image.BILINEAR)
                frame = image_rot.crop((bbox_left, bbox_up, bbox_right, bbox_down))
                # resize
                frame = frame.resize((window_side, window_side), Image.BILINEAR)
                all_frames_in_sequence[f] = np.array(frame)
                
        else:
            bbox_left = full_side // 2 - window_side // 2
            bbox_up = full_side // 2 - window_side // 2
            bbox_right = bbox_left + window_side
            bbox_down = bbox_up + window_side
    
            for f in range(num_frames):
                image_rot = image.rotate(f * speed * direction, resample=Image.BILINEAR)
                frame = image_rot.crop((bbox_left, bbox_up, bbox_right, bbox_down))
                all_frames_in_sequence[f] = np.array(frame)

    elif the_class == 'scale':
        if really_big:
            if direction == 1: # zoom in, things get bigger
                left_x = 0
                up_y = 0
                right_x = full_side
                down_y = full_side
            else: # zoom out, things get smaller
                left_x = full_side// 2 - big_window_side // 2
                up_y = full_side// 2 - big_window_side // 2
                right_x = left_x + big_window_side
                down_y = up_y + big_window_side

            for f in range(num_frames):
                bbox_left = left_x + f * direction * speed
                bbox_right = right_x - f * direction * speed
                bbox_up = up_y + f * direction * speed
                bbox_down = down_y - f * direction * speed

                # print('left: %d, up: %d, right: %d, down: %d' % (bbox_left, bbox_up, bbox_right, bbox_down))

                frame = image.crop((bbox_left, bbox_up, bbox_right, bbox_down))
                frame = frame.resize((window_side, window_side), Image.BILINEAR)

                all_frames_in_sequence[f] = np.array(frame)

        else:
            if direction == 1: # zoom in, things get bigger
                left_x = 0
                up_y = 0
                right_x = full_side
                down_y = full_side
            else: # zoom out, things get smaller
                left_x = full_side // 2 - window_side // 2
                up_y = full_side // 2 - window_side // 2
                right_x = left_x + window_side
                down_y = up_y + window_side
    
            for f in range(num_frames):
                bbox_left = left_x + f * direction * speed
                bbox_right = right_x - f * direction * speed
                bbox_up = up_y + f * direction * speed
                bbox_down = down_y - f * direction * speed
    
                frame = image.crop((bbox_left, bbox_up, bbox_right, bbox_down))
                frame = frame.resize((window_side, window_side), Image.BILINEAR)
    
                all_frames_in_sequence[f] = np.array(frame)

    # save_path_samples = os.path.join(PP.dots_samples, "%s_%d.avi" % (the_class, np.random.randint(1000, 9999)))
    # skvid.vwrite(save_path_samples, all_frames_in_sequence)
    return all_frames_in_sequence, annotation
    

def make_dataset(which, num_samples_per_class, num_frames, side):
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

        for s in tqdm(range(num_samples_per_class)):
            all_frames_array, annotation = generate_sequence_dots(c, num_frames, video_seeds[s], side)
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


# debugging the nvidia dali "Video is too small in at least one dimension"
# make_dataset('val', 100, 30, 33)
# F_ixed: works now with side = 33, instead of 32


# make_dataset('train', 10000, 30, 33)
# make_dataset('val', 2000, 30, 33)
# make_dataset('test', 5000, 30, 33)