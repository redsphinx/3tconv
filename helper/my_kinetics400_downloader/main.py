import time
import numpy as np
import os
import json
from utilities.utils import opt_mkdir, opt_makedirs
import helper.my_kinetics400_downloader.tools as tools
from tqdm import tqdm
import subprocess

'''

TODO
make a list of succeeded downloads

AT INIT
(in case the script abruptly ends we can continue without issues)
make list of downloadeds
remove anything that isn't complete


MODES 
'only_failed': download videos on failed list
'og_list': download things that haven't downloaded yet and that aren't on failed


WHICH
'test', 'train', 'valid'


ONLY_FAILED
create failed_reason file

read failed list
read downloaded list (success)

remove ids that intersect

for the indicated amount of videos:
    download the video
    if success:
        add successful ids on succeeded list
        remove ids from failed
    else:
        get reason of failure
        remove from failed (so that it doesn't try it twice)
        add to failed_reason file


OG_LIST
read og list
read success list
read failed list

make list that removes intersections

for the indicated amount of videos:
    download the video
    if success:
        add successful ids on succeeded list
    else:
        get reason of failure
        add to failed_reason file
    

'''

main_path = tools.main_path
jsons = tools.jsons

# only use once after downloads have crashed
# removes partial files (non .mp4) from downloads
def clean_up_partials():

    print('Removing partial files...')
    cnt = 0
    for which in ['test', 'train', 'valid']:
        download_list = tools.get_downloaded_list(which, full_path=True) # list of strings

        for word in ['part', 'ytdl', '_raw']:
            filtered_download_list = list(filter(lambda x:word in x, download_list))
            for to_be_del in filtered_download_list:
                if os.path.exists(to_be_del):

                    # =========================
                    os.remove(to_be_del)
                    # =========================

                    cnt += 1
    print('%d partial files removed' % cnt)


# only use once after downloads have crashed
# runs a sanity check to find inconsistencies between lists
def crosscheck_lists():
    print('\nCrosschecking lists...')

    # check failed and tbr_failed: if on both, remove from failed
    print('..checking failed and tbr_failed..')
    for which in ['test', 'train', 'valid']:
        tbr_failed_list = set(tools.get_to_be_removed_from_fail_list(which)) # list of ids
        failed_list = set(tools.get_failed_list(which)) # list of ids
        overlap = tbr_failed_list.intersection(failed_list)
        new_failed = list(failed_list - overlap)

        if len(overlap) > 0:
            failed_path = os.path.join(tools.fails, '%s.txt' % which)
            tmp_failed_path = os.path.join(tools.fails, 'tmp_%s.txt' % which)

            with open(tmp_failed_path, 'w') as my_file:
                for vid_id in new_failed:
                    line = '%s\n' % vid_id
                    my_file.write(line)

            print('..removed old path', which, failed_path)

            # =========================
            tools.replace(tmp_failed_path, failed_path)
            # =========================

    print('done')

    # check failed and downloads: if on downloads+failed, remove from failed
    print('..checking downloads and failed..')
    for which in ['test', 'train', 'valid']:
        download_list = set(tools.get_downloaded_list(which, full_path=False)) # list of ids
        failed_list = set(tools.get_failed_list(which)) # list of ids
        overlap = download_list.intersection(failed_list)
        new_failed = list(failed_list - overlap)

        if len(overlap) > 0:
            failed_path = os.path.join(tools.fails, '%s.txt' % which)
            tmp_failed_path = os.path.join(tools.fails, 'tmp_%s.txt' % which)

            with open(tmp_failed_path, 'w') as my_file:
                for vid_id in new_failed:
                    line = '%s\n' % vid_id
                    my_file.write(line)

            print('..removed old path', which, failed_path)

            # =========================
            tools.replace(tmp_failed_path, failed_path)
            # =========================

    print('done')

    # check downloads and success:
    print('..checking downloads and success..')
    for which in ['test', 'train', 'valid']:
        download_list = set(tools.get_downloaded_list(which, full_path=False)) # list of ids
        success_list = set(tools.get_success_list(which)) # list of ids

        # if on success and not download, remove from success
        on_success_not_download = success_list.difference(download_list)
        success_list = success_list - on_success_not_download

        # if on download and not success, add to success
        on_download_not_success = download_list.difference(success_list)
        tmp = success_list.update(on_download_not_success)
        if tmp is not None:
            success_list = list(success_list.update(on_download_not_success))

        if len(on_download_not_success) > 0 or len(on_success_not_download) > 0:
            success_path = os.path.join(tools.successes, '%s.txt' % which)
            tmp_success_path = os.path.join(tools.successes, 'tmp_%s.txt' % which)

            with open(tmp_success_path, 'w') as my_file:
                for vid_id in success_list:
                    line = '%s\n' % vid_id
                    my_file.write(line)

            print('..removed old path', which, success_path)

            # =========================
            tools.replace(tmp_success_path, success_path)
            # =========================

    print('done')

    # check failed and failed_reasons: if on both, remove from failed
    print('..checking failed and failed_reasons..')
    for which in ['test', 'train', 'valid']:
        failed_list = set(tools.get_failed_list(which)) # list of ids
        failed_reasons_list = set(tools.get_failed_reasons_list(which)) # list of ids

        overlap = failed_list.intersection(failed_reasons_list)
        new_failed = list(failed_list - overlap)

        if len(overlap) > 0:
            failed_path = os.path.join(tools.fails, '%s.txt' % which)
            tmp_failed_path = os.path.join(tools.fails, 'tmp_%s.txt' % which)

            with open(tmp_failed_path, 'w') as my_file:
                for vid_id in new_failed:
                    line = '%s\n' % vid_id
                    my_file.write(line)

            print('..removed old path', which, failed_path)

            # =========================
            tools.replace(tmp_failed_path, failed_path)
            # =========================

    print('done')
    print('Finished checking lists')


def make_download_list(mode, which):
    if mode == 'only_failed':
        download_list = tools.get_failed_list(which)

    elif mode == 'og_list':
        total_list = set(tools.get_all_video_ids(which))
        failed_list = set(tools.get_failed_list(which))
        success_list = set(tools.get_success_list(which))

        download_list = list(total_list - failed_list - success_list)

    else:
        download_list = None

    return download_list


def download_videos(vid_id, which, category):
    video_format = 'mp4'
    download_path = os.path.join(tools.main_path, which, category)

    return_code = subprocess.call(
        ["youtube-dl", "https://youtube.com/watch?v={}".format(vid_id), "--quiet", "-f",
         "bestvideo[ext={}]+bestaudio/best".format(video_format), "--output", download_path, "--no-continue"],
        stderr=subprocess.DEVNULL)

    is_success = return_code == 0
    if not is_success:
        opt_reason = str(return_code)
    else:
        opt_reason = None

    return is_success, opt_reason


def add_success(which, vid_id):
    success_list = tools.get_success_list(which)
    success_path = os.path.join(tools.successes, '%s.txt' % which)
    if vid_id not in success_list:

        line = '%s\n' % vid_id
        retry = tools.append_to_file(success_path, line)
        
        while retry:
            time.sleep(0.5)
            retry = tools.append_to_file(success_path, line)



def add_failed_reason(which, vid_id, opt_reason):
    failed_reason_list = tools.get_failed_reasons_list(which)
    failed_reason_path = os.path.join(tools.failed_reasons, '%s.txt' % which)
    if vid_id not in failed_reason_list:

        line = '%s,%s\n' % (vid_id, str(opt_reason))
        retry = tools.append_to_file(failed_reason_path, line)

        while retry:
            time.sleep(0.5)
            retry = tools.append_to_file(failed_reason_path, line)


def add_to_be_removed_from_failed(which, vid_id):
    tbr_fail_list = tools.get_to_be_removed_from_fail_list(which)
    tbr_fail_path = os.path.join(tools.fails, 'tbr_%s.txt' % which)

    if vid_id not in tbr_fail_list:

        line = '%s\n' % (vid_id)
        retry = tools.append_to_file(tbr_fail_path, line)

        while retry:
            time.sleep(0.5)
            retry = tools.append_to_file(tbr_fail_path, line)


def run(mode, which, start, end):
    assert mode in ['only_failed', 'og_list']
    assert which in ['test', 'train', 'valid']

    download_list = make_download_list(mode, which)
    download_list.sort()

    print('=====================================================\n'
          'Downloading mode=%s, which=%s, b=%d, e=%d\n'
          '=====================================================\n'
          % (mode, which, start, end))

    for i in tqdm(range(start, end)):
        video_id = download_list[i]
        is_success, opt_reason = download_videos(video_id, which)

        if is_success:
            add_success(which, video_id)
            if mode == 'only_failed':
                add_to_be_removed_from_failed(which, video_id)
                # remove_from_failed(video_id)
        else:
            assert opt_reason is not None
            add_failed_reason(which, video_id, opt_reason)
            # TODO:
            '''
            if failed reason is 429 then stop
            '''

            if mode == 'only_failed':
                add_to_be_removed_from_failed(which, video_id)
                # remove_from_failed(video_id)


# clean_up_partials()
# crosscheck_lists()
run(mode='only_failed', which='valid', start=0, end=3)
