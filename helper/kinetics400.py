import numpy as np
import os
from utilities.utils import opt_mkdir
import json


main_path = '/fast/gabras/kinetics400_downloader/dataset'
jsons = '/fast/gabras/kinetics400_downloader/resources'
stats_path = '/fast/gabras/kinetics400_downloader/download_stats'
failed_path = '/fast/gabras/kinetics400_downloader/dataset/failed.txt'

opt_mkdir(stats_path)


def get_video_ids(which):
    assert which in ['test', 'train', 'valid']
    if which == 'valid':
        which = 'val'

    src_path = os.path.join(jsons, 'kinetics_%s.json' % which)
    with open(src_path) as json_file:
        data = json.load(json_file)
    keys_list = list(data.keys())
    return keys_list


def get_downloaded(which):
    assert which in ['test', 'train', 'valid']

    which_path = os.path.join(main_path, which)

    # make lists json videos
    all_video_ids = get_video_ids(which)

    # make lists which videos have downloaded
    downloaded_videos = []
    folders = os.listdir(which_path)

    for _f in folders:
        _f_path = os.path.join(which_path, _f)
        vids = os.listdir(_f_path)
        for _v in vids:
            last = _v.split('.')[-1]
            if last == 'mp4':
                downloaded_videos.append(_v.split('.')[0])

    # make lists which videos on failed list
    failed = np.genfromtxt(failed_path, str)

    print('total %s videos: %d\n'
          'successfully downloaded: %d \n'
          'listed as failed: %d' % (which, len(all_video_ids), len(downloaded_videos), len(failed)))


# get_downloaded('train')
# get_downloaded('valid')
# get_downloaded('test')

# total train videos: 246534
# successfully downloaded: 59006
# listed as failed: 222611

# total valid videos: 19906
# successfully downloaded: 4472
# listed as failed: 222611