# This is a toy dataset with 3 classes so that we can validate 3TConv
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


def parameterized_generate_dots(parameters, the_class, direction=None, speed=None, num_frames=30, window_side=32, really_big=False,
                                big_factor=5):

    '''
    parameterized version to let network generate the parameters
    '''

    if the_class == 'translate':
        rad_dot_m_t, num_dots_low_t, num_dots_high_t = parameters
        # 2-10  10-100  20-110
        # 6,70,80

        if speed is None:
            speed = 1

        full_side = window_side + speed * num_frames * 2
        # radius_dot_min = np.random.randint(1, 6)
        radius_dot_min = np.random.randint(1, rad_dot_m_t)
        radius_dot_max = radius_dot_min + 1

        # num_dots_low = speed * 1 / radius_dot_min * 60
        # num_dots_high = speed * 1 / radius_dot_min * 70
        num_dots_low = speed * 1 / radius_dot_min * num_dots_low_t
        num_dots_high = speed * 1 / radius_dot_min * num_dots_high_t

        if really_big:
            full_side = full_side * big_factor
            radius_dot_min = radius_dot_min * big_factor
            radius_dot_max = radius_dot_max * big_factor
            num_dots_low = num_dots_low * big_factor**2
            num_dots_high = num_dots_high * big_factor**2

        num_dots = np.random.randint(num_dots_low, num_dots_high)

    elif the_class == 'rotate':
        rad_dot_m_r, num_dots_low_r, num_dots_high_r = parameters
        # 5,3,30

        full_side = int(np.sqrt(2 * window_side**2)) + 1
        # num_dots = np.random.randint(5, 11)
        num_dots = np.random.randint(num_dots_low_r, num_dots_high_r)
        # radius_dot_min = np.random.randint(1, 4)
        radius_dot_min = np.random.randint(1, rad_dot_m_r)
        radius_dot_max = radius_dot_min + 1

        if really_big:
            full_side = full_side * big_factor
            num_dots = num_dots * big_factor**2
            radius_dot_min = radius_dot_min * big_factor
            radius_dot_max = radius_dot_max * big_factor

    elif the_class == 'scale':
        num_dots_low_s, num_dots_high_s, radius_dot_min_s_pos_dir, radius_dot_max_s_pos_dir, radius_dot_min_s_neg_dir, \
        radius_dot_max_s_neg_dir = parameters
        # 14,15,3,6,6,8

        if speed is None:
            speed = 1
        full_side = window_side + speed * num_frames * 2

        # num_dots = np.random.randint(10, 13)
        num_dots = np.random.randint(num_dots_low_s, num_dots_high_s)
        if direction == 1:
            # radius_dot_min = 6
            # radius_dot_max = 9
            radius_dot_min = radius_dot_min_s_pos_dir
            radius_dot_max = radius_dot_max_s_pos_dir
        else:
            # radius_dot_min = 7
            # radius_dot_max = 10
            radius_dot_min = radius_dot_min_s_neg_dir
            radius_dot_max = radius_dot_max_s_neg_dir

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
    
    # random rotation
    angle = np.random.randint(10, 30)
    direction = np.random.randint(0, 2)
    if direction == 0:
        direction = -1

    image_rot = image.rotate(direction*angle, resample=Image.BILINEAR)
    
    # random scale
    amount = np.random.randint(1, 6)

    if direction == -1:
        frame = image_rot.resize((full_side+amount, full_side+amount), Image.BILINEAR)
        frame = frame.crop((0+amount, 0+amount, full_side-amount, full_side-amount))
    else:
        frame = image_rot.crop((0+amount, 0+amount, full_side-amount, full_side-amount))
        frame = frame.resize((full_side, full_side), Image.BILINEAR)
    
    return frame


def generate_sequence_dots(the_class, num_frames, seed, window_side, really_big=True, big_factor=5, parameters=None):

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

    if parameters is not None:
        image, canvas, full_side = parameterized_generate_dots(parameters, the_class, direction, speed, num_frames,
                                                               really_big=really_big)
    else:
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
    

def make_dataset(which, num_samples_per_class, num_frames, side, parameters=None):

    '''
    rad_dot_m_t, num_dots_low_t, num_dots_high_t = parameters
    rad_dot_m_r, num_dots_low_r, num_dots_high_r = parameters
    num_dots_low_s, num_dots_high_s, radius_dot_min_s_pos_dir, radius_dot_max_s_pos_dir, radius_dot_min_s_neg_dir, radius_dot_max_s_neg_dir = parameters

    parameters = [rad_dot_m_t, num_dots_low_t, num_dots_high_t, rad_dot_m_r, num_dots_low_r, num_dots_high_r,
    num_dots_low_s, num_dots_high_s, radius_dot_min_s_pos_dir, radius_dot_max_s_pos_dir, radius_dot_min_s_neg_dir,
    radius_dot_max_s_neg_dir]
    '''

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


# debugging the nvidia dali "Video is too small in at least one dimension"
# make_dataset('val', 20, 30, 33)
# F_ixed: works now with side = 33, instead of 32


# make_dataset('train', 10000, 30, 33)
# make_dataset('val', 2000, 30, 33)
# make_dataset('test', 5000, 30, 33)

# make_dataset('train', 5000, 30, 33)
# make_dataset('val', 1000, 30, 33)


def config_pv(device_num):
    project_variable = ProjectVariable(debug_mode=True)
    project_variable.end_epoch = 10
    project_variable.dataset = 'dots_frames'
    project_variable.sheet_number = 25
    project_variable.num_in_channels = 1
    project_variable.label_size = 3
    project_variable.label_type = 'categories'
    project_variable.repeat_experiments = 1
    project_variable.save_only_best_run = True
    project_variable.same_training_data = True
    project_variable.randomize_training_data = True
    project_variable.balance_training_data = True
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    project_variable.weight_transform = 'seq'
    project_variable.experiment_state = 'new'
    project_variable.eval_on = 'val'
    project_variable.model_number = 55 # lenet5 2D
    project_variable.experiment_number = 192323838388383
    project_variable.device = device_num
    project_variable.batch_size = 32
    project_variable.batch_size_val_test = 32
    project_variable.inference_only_mode = False
    project_variable.load_model = False # loading model from scratch
    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False
    project_variable.dots_mode = True
    project_variable.stop_at_collapse = False
    project_variable.early_stopping = False
    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.use_adaptive_lr = True
    return project_variable


def opt_remove_dataset():
    if os.path.exists(PP.dots_dataset_avi):
        command = 'rm -rf %s' % PP.dots_dataset_avi
        subprocess.call(command, shell=True)
        command = 'mkdir %s' % PP.dots_dataset_avi
        subprocess.call(command, shell=True)

    if os.path.exists(PP.dots_dataset_frames):
        command = 'rm -rf %s' % PP.dots_dataset_frames
        subprocess.call(command, shell=True)
        command = 'mkdir %s' % PP.dots_dataset_frames
        subprocess.call(command, shell=True)


def sample_params():
    # parameters_path = os.path.join(PP.dots_root, 'resources/parameters_tried.txt')
    # parameters_tried = np.genfromtxt(parameters_path, int, delimiter=',')[:, 0:-1]
    #
    # parameters_tried = list(parameters_tried)

    # 6,70,80   ,5,3,30,    14,15,3,6,6,8,         0.527961
    # 7,70,90   ,4,4,50,    13,14,3,8,6,10,        0.527961

    # randomly choose parameters
    np.random.seed() # reset the seed
    # the_same = True
    # while the_same:
    # rad_dot_m_t = np.random.randint(2, 11) # it 1
    rad_dot_m_t = np.random.randint(5, 8) # it 2
    # rad_dot_m_t = np.arange(2, 11, 1)
    # num_dots_low_t = np.random.randint(1, 11) # it 1
    num_dots_low_t = np.random.randint(6, 9) # it 2
    # num_dots_low_t = np.arange(10, 110, 10)
    # num_dots_high_t = np.random.randint(num_dots_low_t+1, 12) * 10 # it 1
    num_dots_high_t = np.random.randint(num_dots_low_t+1, 11) * 10 # it 2
    num_dots_low_t = num_dots_low_t * 10
    # num_dots_high_t = np.arange(20, 120, 10)

    # 6,70,80,    5,3,30,     14,15,3,6,6,8,      0.527961

    # rad_dot_m_r = np.random.randint(2, 11) # it 1
    rad_dot_m_r = np.random.randint(4, 7) # it 2
    # rad_dot_m_r = np.arange(2, 11, 1)
    # num_dots_low_r = np.random.randint(1, 11) # it 1
    num_dots_low_r = np.random.randint(2, 5) # it 2
    # num_dots_low_r = np.arange(10, 110, 10)
    # num_dots_high_r = np.random.randint(num_dots_low_r+1, 12) * 10 # it 1
    num_dots_high_r = np.random.randint(num_dots_low_r+1, 6) * 10 # it 2
    # num_dots_low_r = num_dots_low_r + 1  # it 2
    num_dots_low_r = num_dots_low_r * 10  # it 3
    # num_dots_high_r = np.arange(20, 120, 10)

    # 6,70,80,    5,3,30,     14,15,3,6,6,8,      0.527961

    # num_dots_low_s = np.random.randint(10, 16) # it 1
    num_dots_low_s = np.random.randint(13, 16) # it 2
    # num_dots_low_s = np.arange(10, 16, 1)
    # num_dots_high_s = np.random.randint(num_dots_low_s+1, 17) # it 1
    num_dots_high_s = np.random.randint(num_dots_low_s+1, 18) # it 2
    # num_dots_high_s = np.arange(11, 17, 1)
    # radius_dot_min_s_pos_dir = np.random.randint(3, 10) # it 1
    radius_dot_min_s_pos_dir = np.random.randint(2, 5) # it 2
    # radius_dot_min_s_pos_dir = np.arange(3, 10, 1)
    # radius_dot_max_s_pos_dir = np.random.randint(radius_dot_min_s_pos_dir+1, 11) # it 1
    radius_dot_max_s_pos_dir = np.random.randint(radius_dot_min_s_pos_dir+1, 9) # it 2
    # radius_dot_max_s_pos_dir = np.arange(4, 11, 1)
    # radius_dot_min_s_neg_dir = np.random.randint(3, 10) # it 1
    radius_dot_min_s_neg_dir = np.random.randint(5, 8) # it 2
    # radius_dot_min_s_neg_dir = np.arange(3, 10, 1)
    # radius_dot_max_s_neg_dir = np.random.randint(radius_dot_min_s_neg_dir+1, 11) # it 1
    radius_dot_max_s_neg_dir = np.random.randint(radius_dot_min_s_neg_dir+1, 11) # it 2
    # radius_dot_max_s_neg_dir = np.arange(4, 11, 1)

        # param_str = '%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d' % (
        # rad_dot_m_t, num_dots_low_t, num_dots_high_t, rad_dot_m_r, num_dots_low_r, num_dots_high_r,
        # num_dots_low_s, num_dots_high_s, radius_dot_min_s_pos_dir, radius_dot_max_s_pos_dir,
        # radius_dot_min_s_neg_dir, radius_dot_max_s_neg_dir)
        #
        # if param_str not in parameters_tried:
    parameters = [rad_dot_m_t, num_dots_low_t, num_dots_high_t, rad_dot_m_r, num_dots_low_r, num_dots_high_r,
                  num_dots_low_s, num_dots_high_s, radius_dot_min_s_pos_dir, radius_dot_max_s_pos_dir,
                  radius_dot_min_s_neg_dir, radius_dot_max_s_neg_dir]
            # the_same = False

    return parameters


def save_results(val_acc, parameters):
    param_path = os.path.join(PP.dots_root, 'resources/parameters_tried.txt')
    param_str = '%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%f\n' % (parameters[0], parameters[1], parameters[2], parameters[3],
                                                            parameters[4], parameters[5], parameters[6], parameters[7],
                                                            parameters[8], parameters[9], parameters[10],
                                                            parameters[11], val_acc)
    with open(param_path, 'a') as my_file:
        my_file.write(param_str)

def generation_loop_with_cnn(device_num=0):

    '''
    rad_dot_m_t, num_dots_low_t, num_dots_high_t = parameters
    2, 10, 20
    rad_dot_m_r, num_dots_low_r, num_dots_high_r = parameters
    2, 10, 20
    num_dots_low_s, num_dots_high_s, radius_dot_min_s_pos_dir, radius_dot_max_s_pos_dir, radius_dot_min_s_neg_dir, radius_dot_max_s_neg_dir = parameters
    10, 11, 3, 4, 3, 4 
    
    '''
    # parameters = [2, 10, 20, 2, 10, 20, 10, 11, 3, 4, 3, 4]

    random_acc = 1/3
    val_acc = 1
    e = 0.05

    it = 1
    while val_acc > random_acc+e:
        opt_remove_dataset()
        parameters = sample_params()
        make_dataset('train', 500, 30, 33, parameters)
        make_dataset('val', 200, 30, 33, parameters)
        pv = config_pv(device_num)
        val_acc = main_file.run(pv)

        if val_acc > random_acc+e:
            print('%d:  val_acc: %f. parameters tried: %s' % (it, val_acc, str(parameters)))
            save_results(val_acc, parameters)
        else:
            print('OPTIMAL PARAMETERS FOUND!')
            print('%d:  val_acc: %f. parameters: %s' % (it, val_acc, str(parameters)))
            save_results(val_acc, parameters)

        it = it + 1


generation_loop_with_cnn(1)


'''
6,70,80,    5,3,30,     14,15,3,6,6,8,      0.527961
'''