# standardizes the size and length of the sequences
from itertools import repeat
import numpy as np
import os
import time
from utilities.utils import opt_mkdir, opt_makedirs
from config import paths as PP
import helper.my_kinetics400_downloader.tools as tools
import helper.my_kinetics400_downloader.main as M
from tqdm import tqdm
# import subprocess
from multiprocessing import Pool
from PIL import Image
import skvideo.io as skvid
# import cv2 as cv
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import math
import random



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

        else:

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


def adjust_frame(np_image, h, w, c):
    # resize to height of h
    if np_image.shape[-1] > 1:
        mode = 'RGB'
    else:
        mode = 'L'

    image = Image.fromarray(np_image, mode=mode)
    or_w, or_h = image.size
    new_w = int(h * or_w / or_h)
    image = image.resize((new_w, h), resample=Image.BICUBIC) # w, h

    if new_w > w:
        delta_w = (new_w - w) // 2
        delta_w_2 = w + delta_w
        image = image.crop((delta_w, 0, delta_w_2, h))  # l, u, r, d
    elif new_w < w:
        delta_w = (w - new_w) // 2
        image = np.array(image)
        pixel_mean = np.mean(np.mean(image, axis=0), axis=0)
        pixel_mean = np.array(pixel_mean, dtype=int)
        canvas = np.ones(shape=(h, w, c), dtype=np.uint8)
        canvas = canvas * pixel_mean
        # paste it
        canvas[:, delta_w:new_w+delta_w, :] = image
        image = canvas

    image = np.array(image, dtype=np.uint8)
    assert image.shape == (h, w, c)

    return image



def standardize_single_video(video_path, height, width, frames, channels=3):
    which = video_path.split('/')[5]
    category = video_path.split('/')[6]
    video_name = video_path.split('/')[-1].split('.')[0]

    cat_path = os.path.join(PP.kinetics400_dataset_150_224, which, category)
    opt_makedirs(cat_path)

    save_path = os.path.join(cat_path, '%s.avi' % video_name)

    if not os.path.exists(save_path):

        if video_path == '/fast/gabras/kinetics400_downloader/dataset/train/drop_kicking/7k7Lj3BR-CM.mp4':
            print('HEREEE')

        video = skvid.vread(video_path)  # (frames, height, width, channels)

        # choose frames
        num_frames = video.shape[0]

        if num_frames < frames:
            missing_frames = frames - num_frames
            copy_number_times = math.ceil(missing_frames / num_frames)

            new_num_frames = num_frames
            frames_to_copy = list(np.arange(0, num_frames))

            while new_num_frames < frames:

                if copy_number_times == 1:
                    extra_frames = list(np.arange(0, missing_frames))
                else:
                    extra_frames = list(np.arange(0, num_frames))

                frames_to_copy.extend(extra_frames)
                frames_to_copy.sort()
                new_num_frames = len(frames_to_copy)

                missing_frames = frames - new_num_frames
                if missing_frames > 0:
                    copy_number_times = math.ceil(missing_frames / num_frames)
                else:
                    try:
                        assert len(frames_to_copy) == frames
                    except AssertionError:
                        print('now we are here')


        elif num_frames > frames:
            if num_frames / frames > 2:

                # copy half of frames, get center
                frames_to_copy = list(np.arange(0, num_frames, 2))
                center = len(frames_to_copy) // 2
                to_copy_half = frames // 2 - 1

                # copy around the denter
                p1 = list(frames_to_copy[center-to_copy_half:center])
                p2 = list(frames_to_copy[center+1:center+1+to_copy_half])

                new_num_frames = len(p1) + len(p2) + 1

                if new_num_frames == frames:
                    frames_to_copy = p1 + frames_to_copy[center] + p2
                else:
                    try:

                        assert frames > new_num_frames
                    except AssertionError:
                        print('here as well')

                    diff = frames - new_num_frames

                    for i in range(diff):
                        if i % 2 == 1:
                            # copy from left
                            cop = frames_to_copy[center-to_copy_half-i]
                            p1.append(cop)

                        else:
                            # copy from right
                            cop = frames_to_copy[center+1+to_copy_half+i]
                            p2.append(cop)

                    frames_to_copy = p1 + [frames_to_copy[center]] + p2

                frames_to_copy.sort()

                # interval = math.floor(num_frames / frames)
                # frames_to_copy = list(np.arange(0, num_frames, interval))
                #
                # copied = len(frames_to_copy)
                #
                # if copied < frames:
                #     # add random frames not previously chosen
                #     diff = frames - copied
                #     bag = list(set(list(np.arange(0, num_frames))) - set(frames_to_copy))
                #     random_indices = random.sample(bag, k=diff)
                #     frames_to_copy.extend(random_indices)
                #     assert len(frames_to_copy) == frames
                #
                # elif len(frames_to_copy) > frames:
                #     # remove random indices
                #     diff = copied - frames
                #     random_indices = random.sample(frames_to_copy, k=diff)
                #     frames_to_copy = list(set(frames_to_copy) - set(random_indices))
                #     assert len(frames_to_copy) == frames
                #
                # else:
                #     assert len(frames_to_copy) == frames

            else:
                frames_to_remove = [n for n in range(0, num_frames, int(math.ceil(num_frames / (num_frames - frames))))]
                leftover = num_frames - len(frames_to_remove)

                if leftover < frames:
                    random_indices = random.sample(frames_to_remove, k=(frames - leftover))
                    for n in random_indices:
                        frames_to_remove.remove(n)

                    assert num_frames - len(frames_to_remove) == frames

                elif leftover > frames:

                    to_add = leftover - frames

                    if to_add == 1:
                        frames_to_remove.append(frames_to_remove[-1]-1)
                    else:
                        selection_list = [i for i in range(num_frames)]
                        tmp = []
                        ind = 0
                        while len(tmp) != num_frames:
                            tmp.append(selection_list.pop(ind))
                            if ind == 0:
                                ind = -1
                            else:
                                ind = 0

                        for i in range(len(tmp)):
                            if tmp[i] not in frames_to_remove:
                                selection_list.append(tmp[i])

                        for a_t in range(to_add):
                            frames_to_remove.append(selection_list[a_t])

                frames_to_remove.sort()

                frames_to_copy = list(np.arange(0, num_frames))
                for n in frames_to_remove:
                    frames_to_copy.remove(n)

                assert len(frames_to_copy) == frames

        else:
            frames_to_copy = list(np.arange(0, num_frames))

        frames_to_copy.sort()

        # collect the relevant frames
        video = video[frames_to_copy]
        new_video = np.zeros(shape=(frames, height, width, channels), dtype=np.uint8)

        for i in range(frames):
            new_video[i] = adjust_frame(video[i], height, width, channels)

        skvid.vwrite(save_path, new_video)


def get_videos(which):

    saved_list_path = os.path.join(tools.resources, '%s_to_be_standardized.txt' % which)

    if os.path.exists(saved_list_path):
        all_videos = np.genfromtxt(saved_list_path, str)

    else:

        which_path = os.path.join(tools.main_path, which)
        categories = os.listdir(which_path)
        categories.sort()
        all_videos = []

        with open(saved_list_path, 'a') as my_file:
            for c in categories:
                cat_path = os.path.join(which_path, c)
                stat_path = os.path.join(cat_path, 'stats.txt')
                stats = np.genfromtxt(stat_path, str, delimiter=',')
                videos = list(stats[:,0])
                videos = [os.path.join(cat_path, '%s.mp4' % i) for i in videos]
                for v in videos:
                    line = '%s\n' % v
                    my_file.write(line)

                all_videos.extend(videos)

    all_videos.sort()
    return all_videos


def standardize_dataset(which, b, e, height, width, frames, parallel=False, num_processes=20):
    '''
    h x w:  150 x 224
    num_frames = 30, 60
    save as avi
    '''

    all_videos = get_videos(which)

    if b is None:
        b = 0
    if e is None:
        e = len(all_videos)

    all_videos = all_videos[b:e]

    if not parallel:
        for vid_path in tqdm(all_videos):
            standardize_single_video(vid_path, height, width, frames)

    else:
        pool = Pool(processes=num_processes)
        pool.apply_async(standardize_single_video)
        pool.starmap(standardize_single_video, zip(all_videos, repeat(height), repeat(width), repeat(frames)))


# standardize_dataset('train', 10, 100, 150, 224, 30, parallel=False)
# st = time.time()
b = 40300   # broke
e = 70300
# standardize_dataset('train', b, e, 150, 224, 30, parallel=True, num_processes=30)  # broke
standardize_dataset('train', b, e, 150, 224, 30, parallel=False)  # broke


# en = time.time()
# tot = (en - st) / 60
# print('standardized %d videos in %f minutes' % (e-b, tot))

