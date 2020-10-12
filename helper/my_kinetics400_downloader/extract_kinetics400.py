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


def aggregate_stats(which, save_fig=True):

    def save_figure(var, var_count, title, save_path):
        fig, ax = plt.subplots(figsize=(8,8))

        max_bin = 20
        if len(var) >= max_bin:
            var = var[:max_bin]
            var_count = var_count[:max_bin]

        p1 = ax.bar(var, var_count)
        ax.set_ylabel('count')
        ax.set_title(title)
        # xticks = ['%d-%d' % (i - var[0], i) for i in var]
        # plt.xticks(var, xticks)
        plt.xticks(np.arange(len(var)), var, rotation='vertical')

        # for plot in p1:-
        #     height = plot.get_height()
        #     ax.annotate('{}'.format(height),
        #                 xy=(plot.get_x() + plot.get_width() / 2, height),
        #                 xytext=(0, 3),  # 3 points vertical offset
        #                 textcoords="offset points",
        #                 ha='center', va='bottom')
        plt.savefig(save_path)

    def add_to_dict(which_dict, which_list):
        uniq = list(set(which_list))
        uniq.sort()
        for i in uniq:
            try:
                which_dict[i] = which_dict[i] + which_list.count(i)
            except KeyError:
                which_dict[i] = which_list.count(i)

        return which_dict


    which_path = os.path.join(tools.main_path, which)
    cat_folders = os.listdir(which_path)
    cat_folders.sort()

    stats_orientation = {'landscape':0, 'portrait':0, 'square':0}
    stats_frames = {}
    stats_h_w = {}

    for cf in cat_folders:
        cat_stat_path = os.path.join(which_path, cf, 'stats.txt')
        cat_stat = np.genfromtxt(cat_stat_path, 'str', delimiter=',')
        h_w = cat_stat[:, 1:3]
        frames = list(cat_stat[:, 3])
        orientation = list(cat_stat[:, 4])

        new_h_w = ['%s,%s' % (h_w[i, 0], h_w[i, 1]) for i in range(len(h_w))]
        stats_h_w = add_to_dict(stats_h_w, new_h_w)
        stats_frames = add_to_dict(stats_frames, frames)
        stats_orientation = add_to_dict(stats_orientation, orientation)

    names = ['stats_h_w', 'stats_frames', 'stats_orientation']
    for i, which_stats in enumerate([stats_h_w, stats_frames, stats_orientation]):
        stats = list(which_stats.keys())
        stats.sort()
        stats_count = [which_stats[j] for j in stats]

        zipped = list(zip(stats_count, stats))
        zipped.sort(reverse=True)
        stats_count, stats = zip(*zipped)

        print('%s Top 3 highest:' % names[i])
        for j in range(3):
            print('%s   %s' % (stats[j], stats_count[j]))

        if names[i] == 'stats_h_w':
            h_w = np.genfromtxt(stats, int, delimiter=',')
            _h = list(h_w[:, 0])
            _w = list(h_w[:, 1])
            zipped = list(zip(_h, _w))
            zipped.sort()
            _h, _w = zip(*zipped)
            print('min h: %d x %d\n'
                  'max h: %d x %d' % (_h[0], _w[0], _h[-1], _w[-1]))
            zipped = list(zip(_w, _h))
            zipped.sort()
            _w, _h = zip(*zipped)
            print('min w: %d x %d\n'
                  'max w: %d x %d' % (_h[0], _w[0], _h[-1], _w[-1]))

            h_w = np.genfromtxt(stats, int, delimiter=',')
            _h = h_w[:, 0]
            _w = h_w[:, 1]
            avg_h = int(np.sum(np.array(stats_count) * _h) / np.sum(np.array(stats_count)))
            avg_w = int(np.sum(np.array(stats_count) * _w) / np.sum(np.array(stats_count)))

            print('avg h: %d, avg w: %d' % (avg_h, avg_w))

        if names[i] == 'stats_frames':
            stats = np.array(stats).astype(int)
            print('min: %d, max: %d' % (stats.min(), stats.max()))
            avg_frames = int(np.sum(np.array(stats_count) * stats) / np.sum(np.array(stats_count)))
            print('avg frames: %d' % (avg_frames))


        if save_fig:
            s_path = os.path.join(tools.resources, '%s.jpg' % names[i])
            save_figure(stats, stats_count, names[i], s_path)

# aggregate_stats('train', save_fig=False)
# stats_h_w Top 3 highest:
# 720,1280   41603
# 480,640   39936
# 1080,1920   37213
# min h: 56 x 224
# max h: 2160 x 3840
# min w: 144 x 82
# max w: 2160 x 3840
# avg h: 631, avg w: 944
# stats_frames Top 3 highest:
# 300   117664
# 250   25483
# 240   11245
# min: 3, max: 600
# avg frames: 265
# stats_orientation Top 3 highest:
# landscape   187562
# portrait   24337
# square   1909

def standardize_dataset():
    '''
    h x w:  150 x 224
    num_frames = 30, 60
    save as avi


    '''

    pass