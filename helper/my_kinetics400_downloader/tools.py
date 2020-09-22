from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import subprocess
from filelock import FileLock
import numpy as np
import os
from utilities.utils import opt_mkdir
import json
# import helper.my_kinetics400_downloader.main as M


main_path = '/fast/gabras/kinetics400_downloader/dataset'
resources = '/fast/gabras/kinetics400_downloader/resources'
# stats_path = '/fast/gabras/kinetics400_downloader/download_stats'
fails = '/fast/gabras/kinetics400_downloader/fails'
successes = '/fast/gabras/kinetics400_downloader/successes'
failed_reasons = '/fast/gabras/kinetics400_downloader/failed_reasons'

og_failed_path = '/fast/gabras/kinetics400_downloader/dataset/failed.txt'
# tmp_failed_path = '/fast/gabras/kinetics400_downloader/dataset/tmp_failed.txt'
# success_path = '/fast/gabras/kinetics400_downloader/dataset/success.txt'
# tmp_success_path = '/fast/gabras/kinetics400_downloader/dataset/tmp_success.txt'

download_plots = '/fast/gabras/kinetics400_downloader/plots_download_progress'


# opt_mkdir(stats_path)
# opt_mkdir(fails)
# opt_mkdir(successes)
# opt_mkdir(failed_reasons)


def get_all_video_ids(which):
    assert which in ['test', 'train', 'valid']
    if which == 'valid':
        which = 'val'

    src_path = os.path.join(resources, 'kinetics_%s.json' % which)
    with open(src_path) as json_file:
        data = json.load(json_file)
    keys_list = list(data.keys())
    return keys_list

# the_list = get_all_video_ids('train')
# print(len(the_list))

def get_downloaded_list(which, full_path=False):
    which_path = os.path.join(main_path, which)
    folders = os.listdir(which_path)
    download_list = []

    if full_path:
        for _f in folders:
            _f_path = os.path.join(which_path, _f)
            vids = os.listdir(_f_path)
            for _v in vids:
                full_vid_path = os.path.join(_f_path, _v)
                download_list.append(full_vid_path)
    else: # only ids
        for _f in folders:
            _f_path = os.path.join(which_path, _f)
            vids = os.listdir(_f_path)
            for _v in vids:
                _v = _v.split('.')[0]
                download_list.append(_v)

    return download_list


def fix_failed_list():
    og_failed = set(np.genfromtxt(og_failed_path, str))

    for which in ['test', 'train', 'valid']:
        which_failed_path = os.path.join(fails, '%s.txt' % which)
        which_ids = set(get_all_video_ids(which))

        overlap = list(og_failed.intersection(which_ids))
        with open(which_failed_path, 'a') as my_file:
            for id in overlap:
                line = '%s\n' % id
                my_file.write(line)




def get_failed_list(which):
    
    failed_path = os.path.join(fails, '%s.txt' % which)
    if os.path.exists(failed_path):
        return list(np.genfromtxt(failed_path, str))
    else:
        with open(failed_path, 'w') as my_file:
            print('created file: %s' % failed_path)
        return []


def get_failed_reasons_list(which):
    failed_reasons_path = os.path.join(failed_reasons, '%s.txt' % which)
    if os.path.exists(failed_reasons_path):

        def get_list():
            the_list = None
            err = None
            try:
                the_list = np.genfromtxt(failed_reasons_path, str, delimiter=',', skip_header=True)
                retr = False
            except ValueError or TimeoutError or AttributeError as err_:
                retr = True
                err = err_

            return retr, the_list, err

        def make_new_list():
            try:
                lock_path = failed_reasons_path.split('.txt')[0] + '.lock'
                lock = FileLock(lock_path, timeout=1)
                with lock:
                    new_failed_reasons_path = os.path.join(failed_reasons, '%s_new.txt' % which)
                    command = "cat %s | head -n -1 > %s" % (failed_reasons_path, new_failed_reasons_path)
                    subprocess.call(command, shell=True)
                    replace(new_failed_reasons_path, failed_reasons_path)
            except TimeoutError:
                # another process is trying to correct it already
                time.sleep(1)


        retry, failed_reasons_list, error = get_list()

        if not retry:
            return failed_reasons_list

        elif retry and type(error) == TimeoutError:
            while retry:
                time.sleep(1)
                retry, failed_reasons_list, error = get_list()
                if not retry:
                    return failed_reasons_list

        elif retry and type(error) == ValueError:
            make_new_list()

        elif retry and type(error) == AttributeError:
            while retry:
                make_new_list()
                retry, failed_reasons_list, error = get_list()

        retry, failed_reasons_list, error = get_list()
        if not retry:
            return failed_reasons_list

        elif retry and type(error) == TimeoutError:
            while retry:
                retry, failed_reasons_list, error = get_list()
                if not retry:
                    return failed_reasons_list

    else:
        with open(failed_reasons_path, 'w') as my_file:
            line = 'id,reason\n'
            my_file.write(line)
            print('created file: %s' % failed_reasons_path)
        return np.array([])

    
def get_success_list(which):
    success_path = os.path.join(successes, '%s.txt' % which)
    if os.path.exists(success_path):
        return list(np.genfromtxt(success_path, str))
    else:
        with open(success_path, 'w') as my_file:
            print('created file: %s' % success_path)
        return []


def replace(p1, p2):
    # replaces p2 with p1
    # p1 = new file
    # p2 = old file
    # example:
    # replace(tmp_failed_path, failed_path)
    # os.remove(failed_path)
    # os.rename(tmp_failed_path, failed_path)
    # os.remove(tmp_failed_path)

    assert os.path.exists(p1)
    if os.path.exists(p2):
        os.remove(p2)
    os.rename(p1, p2)



def append_to_file(file_path, line):

    if type(line) == str:
        lock_path = file_path.split('.txt')[0] + '.lock'

        try:
            lock = FileLock(lock_path, timeout=1)
            with lock:
                with open(file_path, 'a') as my_file:
                    my_file.write(line)

                retry = False
            # with open(file_path, 'a') as my_file:wh
            #     my_file.write(line)

        # except IOError:
        except TimeoutError:
            retry = True

        return retry

    elif type(line) == list:
        with open(file_path, 'a') as my_file:
            for i in line:
                _l = '%s\n' % i
                my_file.write(_l)



def get_to_be_removed_from_fail_list(which):
    tbr_fail_path = os.path.join(fails, 'tbr_%s.txt' % which)
    if os.path.exists(tbr_fail_path):
        tmp = np.genfromtxt(tbr_fail_path, str)
        if tmp.shape == ():
            return [str(tmp)]
        else:
            return list(tmp)
    else:
        with open(tbr_fail_path, 'w') as my_file:
            print('created file: %s' % tbr_fail_path)
        return []


def fix_category_text(name):
    name = name.replace(' ', '_')
    return name


def get_category(which, vid_id):
    if which == 'valid':
        which = 'val'
    src_path = os.path.join(resources, 'kinetics_%s.json' % which)
    with open(src_path) as json_file:
        data = json.load(json_file)
    category = data[vid_id]['annotations']['label']
    category = fix_category_text(category)
    return category


def get_clip_times(which, vid_id):
    if which == 'valid':
        which = 'val'
    src_path = os.path.join(resources, 'kinetics_%s.json' % which)
    with open(src_path) as json_file:
        data = json.load(json_file)
    times = data[vid_id]['annotations']['segment']
    start = times[0]
    end = times[1]
    return start, end


def write_new_file(which, new_file_path, old_file_path, the_new_list):

    if 'failed_reason' in new_file_path:
        assert ',' in the_new_list[0]

    with open(new_file_path, 'w') as my_file:
        if 'failed_reason' in new_file_path:
            line = 'id,reason\n'
            my_file.write(line)

        for _i in the_new_list:
            thing = _i.strip()
            line = '%s\n' % thing
            my_file.write(line)

    print('..removed old path', which, old_file_path)

    # =========================
    replace(new_file_path, old_file_path)
    # =========================


def get_total_per_category_list(which):
    assert which in ['test', 'train', 'valid']
    num_categories = 400
    
    vid_categories_path = os.path.join(resources, '%s_vids_categories.txt' % which)
    
    if not os.path.exists(vid_categories_path):
        which_path = os.path.join(main_path, which)
        all_categories = os.listdir(which_path)
        all_categories.sort()

        total = [0]*num_categories
        total_ids = get_all_video_ids(which)

        if which == 'valid':
            which = 'val'
        src_path = os.path.join(resources, 'kinetics_%s.json' % which)
        with open(src_path) as json_file:
            data = json.load(json_file)

        for vid in tqdm(total_ids):
            cat = data[vid]['annotations']['label']
            cat = fix_category_text(cat)

            _ind = all_categories.index(cat)
            total[_ind] = total[_ind] + 1

        # write to file
        with open(vid_categories_path, 'w') as my_file:
            for i in range(num_categories):
                line = '%s,%d\n' % (all_categories[i], total[i])
                my_file.write(line)

    total_final = np.genfromtxt(vid_categories_path, str, delimiter=',', skip_header=False)

    return total_final


def download_progress_per_class(which):
    assert which in ['test', 'train', 'valid']

    path = os.path.join(main_path, which)
    all_categories = os.listdir(path)
    all_categories.sort()

    num_categories = len(all_categories)

    current_count = []
    for i in range(num_categories):
        cat = all_categories[i]
        num_vids_in_cat = len(os.listdir(os.path.join(path, cat)))
        current_count.append(num_vids_in_cat)

    total = get_total_per_category_list(which)

    bins = list(np.arange(10, 110, 10))
    bin_count = [0]*len(bins)

    for i in range(num_categories):
        ratio = current_count[i] / int(total[i, 1]) * 100

        for j, v in enumerate(bins):
            if ratio < v:
                bin_count[j] = bin_count[j] + 1
                break

    fig, ax = plt.subplots()
    p1 = ax.bar(bins, bin_count)
    ax.set_ylim(0, 400)
    ax.set_ylabel('classes (400 total)')
    ax.set_xlabel('percentage successful downloads')
    ax.set_title('%s successful downloads across classes' % which)
    xticks = ['%d-%d' % (i - bins[0], i) for i in bins]
    plt.xticks(bins, xticks)

    for plot in p1:
        height = plot.get_height()
        ax.annotate('{}'.format(height),
                    xy=(plot.get_x() + plot.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        # ax.text(v + 3, i + .25, str(v))

    current_date = time.strftime("%Y_%m_%d_%H_%M_%S")
    save_location = os.path.join(download_plots, '%s_%s.jpg' % (which, current_date))
    plt.savefig(save_location)


# get_downloaded('train')
# get_downloaded('valid')
# get_downloaded('test')

# total train videos: 246534
# successfully downloaded: 59006
# listed as failed: 222611

# total valid videos: 19906
# successfully downloaded: 4472
# listed as failed: 222611

# get_total_per_category_list('train')
# download_progress_per_class('train')