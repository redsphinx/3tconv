import numpy as np
import os
from utilities.utils import opt_mkdir
import json


main_path = '/fast/gabras/kinetics400_downloader/dataset'
jsons = '/fast/gabras/kinetics400_downloader/resources'
# stats_path = '/fast/gabras/kinetics400_downloader/download_stats'
fails = '/fast/gabras/kinetics400_downloader/fails'
successes = '/fast/gabras/kinetics400_downloader/successes'
failed_reasons = '/fast/gabras/kinetics400_downloader/failed_reasons'

og_failed_path = '/fast/gabras/kinetics400_downloader/dataset/failed.txt'
# tmp_failed_path = '/fast/gabras/kinetics400_downloader/dataset/tmp_failed.txt'
# success_path = '/fast/gabras/kinetics400_downloader/dataset/success.txt'
# tmp_success_path = '/fast/gabras/kinetics400_downloader/dataset/tmp_success.txt'

# opt_mkdir(stats_path)
# opt_mkdir(fails)
# opt_mkdir(successes)
# opt_mkdir(failed_reasons)


def get_all_video_ids(which):
    assert which in ['test', 'train', 'valid']
    if which == 'valid':
        which = 'val'

    src_path = os.path.join(jsons, 'kinetics_%s.json' % which)
    with open(src_path) as json_file:
        data = json.load(json_file)
    keys_list = list(data.keys())
    return keys_list


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


# fix_failed_list()


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
        return list(np.genfromtxt(failed_reasons_path, str, delimiter=',', skip_header=True))
    else:
        with open(failed_reasons_path, 'w') as my_file:
            line = 'id,reason\n'
            my_file.write(line)
            print('created file: %s' % failed_reasons_path)
        return []

    
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
    assert os.path.exists(p1)
    # assert os.path.exists(p2)
    if os.path.exists(p2):
        os.remove(p2)
    os.rename(p1, p2)
    # os.remove(p1)

    # example:
    # replace(tmp_failed_path, failed_path)
    # os.remove(failed_path)
    # os.rename(tmp_failed_path, failed_path)
    # os.remove(tmp_failed_path)


def append_to_file(file_path, line):
    try:
        with open(file_path, 'a') as my_file:
            my_file.write(line)
            retry = False
    except IOError:
        retry = True

    return retry


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
    src_path = os.path.join(jsons, 'kinetics_%s.json' % which)
    with open(src_path) as json_file:
        data = json.load(json_file)
    category = data[vid_id]['annotations']['label']
    category = fix_category_text(category)
    return category


def get_clip_times(which, vid_id):
    if which == 'valid':
        which = 'val'
    src_path = os.path.join(jsons, 'kinetics_%s.json' % which)
    with open(src_path) as json_file:
        data = json.load(json_file)
    times = data[vid_id]['annotations']['segment']
    start = times[0]
    end = times[1]
    return start, end



# get_downloaded('train')
# get_downloaded('valid')
# get_downloaded('test')

# total train videos: 246534
# successfully downloaded: 59006
# listed as failed: 222611

# total valid videos: 19906
# successfully downloaded: 4472
# listed as failed: 222611
