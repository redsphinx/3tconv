# standardizes the size and length of the sequences
import numpy as np
import os
import time
from utilities.utils import opt_mkdir, opt_makedirs
import helper.my_kinetics400_downloader.tools as tools
import helper.my_kinetics400_downloader.main as M
from tqdm import tqdm
import subprocess
from multiprocessing import Pool
from PIL import Image
import skvideo.io as skvid
import cv2 as cv
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt



def get_info(meta_data):
    height = meta_data['@height']
    width = meta_data['@width']
    frames = meta_data['@nb_frames']
    retry = False
    return height, width, frames, retry


def get_single_statistics(video_path):
    try:
        meta_data = skvid.ffprobe(video_path)['video']
    except KeyError:
        return False, None, None, None

    retry = True
    while retry:
        try:
            height, width, frames, retry = get_info(meta_data)
        except KeyError:
            height, width, frames, retry = get_info(meta_data)

    return True, int(height), int(width), int(frames)


def get_orientation(h, w):
    ratio = w / h

    if ratio > 1.2:
        return 'landscape'
    elif ratio < 0.8:
        return 'portrait'
    else:
        return 'square'



def single_video(video_path):

    if os.path.exists(video_path):

        path_info = video_path.split('/')
        video_id = path_info[-1].split('.')[0]
        which = path_info[5]
        category = path_info[6]

        has_video, height, width, frames = get_single_statistics(video_path)

        if has_video:
            orientation = get_orientation(height, width)
            # stats = '%s,%d,%d,%d,%s' % (video_id, height, width, frames, orientation)
            stats = [video_id, height, width, frames, orientation]
            M.add_category_statistic(which, category, stats)

            # print(stats)
            # print('%s statistics:\n'
            #       'height: %d\n'
            #       'width: %d\n'
            #       'frames: %d\n'
            #       'orientation: %s' % (video_path, height, width, frames, orientation))

        else:
            # delete the file
            # =========================
            # os.remove(video_path)
            # =========================
            # print('ONLY AUDIO: ', video_path)

            # add it to failed reason
            M.add_failed_reason(which, video_id, 'only_audio')


def all_videos(which, start, end, parallel=False, num_processes=10):
    all_video_paths = tools.get_all_video_paths(which)

    if start is None:
        start = 0
    if end is None:
        end = len(all_video_paths)

    all_video_paths.sort()
    all_video_paths = all_video_paths[start:end]

    if parallel:
        pool = Pool(processes=num_processes)
        pool.apply_async(single_video)
        pool.map(single_video, all_video_paths)

    else:
        for p in all_video_paths:
            single_video(p)


beg = time.time()
print('getting information...')
all_videos('train', 0, None, parallel=True, num_processes=20)
# all_videos('train', 0, 3)
the_end = time.time()

duration = the_end - beg
print(duration, 'sec')


# p_only_audio = '/fast/gabras/kinetics400_downloader/dataset/train/playing_poker/TaJmMqa4k9M.mp4'
# p_square = '/fast/gabras/kinetics400_downloader/dataset/train/hoverboarding/U41Gtikqg98.mp4'
# p_portrait = '/fast/gabras/kinetics400_downloader/dataset/train/hoverboarding/OuxEfxIJLUk.mp4'
# p_landscape= '/fast/gabras/kinetics400_downloader/dataset/train/playing_poker/Ba2AlQktRRM.mp4'
