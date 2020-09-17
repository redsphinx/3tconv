from itertools import repeat
import time
from multiprocessing import Pool
import os
from utilities.utils import opt_mkdir, opt_makedirs
import helper.my_kinetics400_downloader.tools as tools
from tqdm import tqdm
import subprocess


main_path = tools.main_path
resources = tools.resources

# only use once after downloads have crashed
# removes partial files (non .mp4) from downloads
def clean_up_partials():

    print('Removing partial files...')
    cnt = 0
    for which in ['train', 'valid']:
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
    for which in ['train', 'valid']:
        tbr_failed_list = set(tools.get_to_be_removed_from_fail_list(which)) # list of ids
        failed_list = set(tools.get_failed_list(which)) # list of ids
        overlap = tbr_failed_list.intersection(failed_list)
        new_failed = list(failed_list - overlap)

        if len(overlap) > 0:
            failed_path = os.path.join(tools.fails, '%s.txt' % which)
            tmp_failed_path = os.path.join(tools.fails, 'tmp_%s.txt' % which)

            tools.write_new_file(which, tmp_failed_path, failed_path, new_failed)

    print('done')

    # check failed and downloads: if on downloads+failed, remove from failed
    print('..checking downloads and failed..')
    for which in ['train', 'valid']:
        download_list = set(tools.get_downloaded_list(which, full_path=False)) # list of ids
        failed_list = set(tools.get_failed_list(which)) # list of ids
        overlap = download_list.intersection(failed_list)
        new_failed = list(failed_list - overlap)

        if len(overlap) > 0:
            failed_path = os.path.join(tools.fails, '%s.txt' % which)
            tmp_failed_path = os.path.join(tools.fails, 'tmp_%s.txt' % which)

            tools.write_new_file(which, tmp_failed_path, failed_path, new_failed)

    print('done')

    # check downloads and success:
    print('..checking downloads and success..')
    for which in ['train', 'valid']:
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

            tools.write_new_file(which, tmp_success_path, success_path, success_list)

    print('done')

    # check failed and failed_reasons: if on both, remove from failed
    print('..checking failed and failed_reasons..')
    for which in ['train', 'valid']:
        failed_list = set(tools.get_failed_list(which)) # list of ids
        failed_reasons_list = tools.get_failed_reasons_list(which)
        
        if len(failed_reasons_list.shape) == 1:
            failed_reasons_list = list(failed_reasons_list)
        else:
            failed_reasons_list = list(failed_reasons_list[:,0])
        failed_reasons_list = set(failed_reasons_list) # list of ids

        overlap = failed_list.intersection(failed_reasons_list)
        new_failed = list(failed_list - overlap)

        if len(overlap) > 0:
            failed_path = os.path.join(tools.fails, '%s.txt' % which)
            tmp_failed_path = os.path.join(tools.fails, 'tmp_%s.txt' % which)

            tools.write_new_file(which, tmp_failed_path, failed_path, new_failed)

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


def download_videos(vid_id, which):
    video_format = 'mp4'
    category = tools.get_category(which, vid_id)

    save_folder_path = os.path.join(tools.main_path, which, category)
    opt_makedirs(save_folder_path)
    raw_video_path = os.path.join(save_folder_path, '%s_raw.mp4' % vid_id)
    slice_path = os.path.join(save_folder_path, '%s.mp4' % vid_id)

    # check if another process is already downloading it or if it has already been downloaded
    if os.path.exists(raw_video_path):
        is_success = False
        opt_reason = 'raw_exists'
        return is_success, opt_reason
    elif os.path.exists(slice_path):
        is_success = False
        opt_reason = 'download_complete'
        return is_success, opt_reason

    # HERE
    cookies_path = '/fast/gabras/kinetics400_downloader/cookies.txt'
    # HERE

    download_command = "youtube-dl https://youtube.com/watch?v=%s --cookies %s --quiet -f bestvideo[ext=%s]+bestaudio/best --output %s --no-continue" \
                       % (vid_id, cookies_path, video_format, raw_video_path)
    download_proc = subprocess.Popen(download_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    stdout, stderr = download_proc.communicate()
    stderr = stderr.decode('utf-8')
    stderr = stderr.strip()
    if 'mkv' in stderr:
        raw_video_path = os.path.join(save_folder_path, '%s_raw.mkv' % vid_id)

    if download_proc.returncode == 0:
        download_success = True
        opt_reason = None
    else:
        download_success = False
        opt_reason = stderr

    # cut at indicated times
    if download_success:
        clip_start, clip_end = tools.get_clip_times(which, vid_id)
        cut_command = "ffmpeg -loglevel quiet -i %s -strict -2 -ss %f -to %f %s" \
                           % (raw_video_path, clip_start, clip_end, slice_path)
        cut_proc = subprocess.Popen(cut_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    
        stdout, stderr = cut_proc.communicate()

        if cut_proc.returncode == 0:
            cut_success = True
            opt_reason = None
        else:
            cut_success = False
            opt_reason = stderr.decode('utf-8')

        if cut_success:
            assert os.path.exists(raw_video_path) and os.path.exists(slice_path)

            # =========================
            os.remove(raw_video_path)
            # =========================

        cut_proc.kill()
            
    else:
        cut_success = False

    if download_success and cut_success:
        is_success = True
    else:
        is_success = False

    download_proc.kill()
    del(stderr)
    del(stdout)

    return is_success, opt_reason


def add_success(which, vid_id):
    success_list = tools.get_success_list(which)
    success_path = os.path.join(tools.successes, '%s.txt' % which)
    if vid_id not in success_list:

        line = '%s\n' % vid_id
        retry = tools.append_to_file(success_path, line)
        
        while retry:
            time.sleep(1)
            retry = tools.append_to_file(success_path, line)


def add_failed_reason(which, vid_id, opt_reason):
    # failed_reason_list = tools.get_failed_reasons_list(which) # array, 2 x len

    failed_reason_path = os.path.join(tools.failed_reasons, '%s.txt' % which)
    failed_reason_list = tools.get_failed_reasons_list(which)
    if len(failed_reason_list.shape) == 1:
        failed_reason_list = list(failed_reason_list)
    else:
        failed_reason_list = list(failed_reason_list[:,0])

    if not vid_id in failed_reason_list:
    # if not vid_id in failed_reason_list:

        line = '%s,%s\n' % (vid_id, str(opt_reason))
        assert ',' in line
        retry = tools.append_to_file(failed_reason_path, line)

        while retry:
            time.sleep(1)
            retry = tools.append_to_file(failed_reason_path, line)


def add_to_be_removed_from_failed(which, vid_id):
    tbr_fail_list = tools.get_to_be_removed_from_fail_list(which)
    tbr_fail_path = os.path.join(tools.fails, 'tbr_%s.txt' % which)

    if vid_id not in tbr_fail_list:

        line = '%s\n' % (vid_id)
        retry = tools.append_to_file(tbr_fail_path, line)

        while retry:
            time.sleep(1)
            retry = tools.append_to_file(tbr_fail_path, line)


# check the failed_reasons_list for 429 errors; put the ids back on fail_list, remove them from failed_reasons_list
def clean_up_errors_fr_list(error_type):
    # error_type = str, for example '429' or 'mkv'
    for which in ['train', 'valid']:
        failed_reasons_list = tools.get_failed_reasons_list(which)
        fail_list = tools.get_failed_list(which)

        to_be_added_to_failed_list = []
        to_keep_failed_reasons = []

        fr_vid_ids = failed_reasons_list[:, 0]
        reasons = failed_reasons_list[:, 1]
        for _r in range(len(reasons)):
            if error_type in reasons[_r]:
                if not fr_vid_ids[_r] in fail_list:
                    to_be_added_to_failed_list.append(fr_vid_ids[_r])

            else:
                to_keep_failed_reasons.append('%s,%s' % (fr_vid_ids[_r], reasons[_r]))

        failed_path = os.path.join(tools.fails, '%s.txt' % which)
        tools.append_to_file(failed_path, to_be_added_to_failed_list)

        old_failed_reasons_path = os.path.join(tools.failed_reasons, '%s.txt' % which)
        new_failed_reasons_path = os.path.join(tools.failed_reasons, 'tmp_%s.txt' % which)
        tools.write_new_file(which, new_failed_reasons_path, old_failed_reasons_path, to_keep_failed_reasons)


# clean_up_errors_fr_list('429')


def run(mode, which, start, end):
    assert mode in ['only_failed', 'og_list']
    assert which in ['train', 'valid']

    download_list = make_download_list(mode, which)
    download_list.sort()

    print('=====================================================\n'
          'Downloading mode=%s, which=%s, b=%d, e=%d\n'
          '=====================================================\n'
          % (mode, which, start, end))

    num_success = 0
    num_fails = 0
    for i in tqdm(range(start, end)):
        video_id = download_list[i]
        is_success, opt_reason = download_videos(video_id, which)

        if is_success:
            add_success(which, video_id)
            num_success += 1
            if mode == 'only_failed':
                add_to_be_removed_from_failed(which, video_id)
        else:
            assert opt_reason is not None
            num_fails += 1

            if '429' in opt_reason:
                print('Successes: %d\n'
                      'Failures: %d' % (num_success, num_fails))
                print('\n'
                      'ERROR 429 ENCOUNTERED. PROCESS WILL TERMINATE NOW.'
                      '\n')
                return
            else:
                add_failed_reason(which, video_id, opt_reason)
                if mode == 'only_failed':
                    add_to_be_removed_from_failed(which, video_id)



    print('Successes: %d\n'
          'Failures: %d' % (num_success, num_fails))


def single_run(video_id, mode, which):
    is_success, opt_reason = download_videos(video_id, which)

    if is_success:
        add_success(which, video_id)
        if mode == 'only_failed':
            add_to_be_removed_from_failed(which, video_id)
        print('Success: %s' % video_id)
    else:
        assert opt_reason is not None

        if '429' in opt_reason:
            code = 429
            return code
        elif 'cookie' in opt_reason:
            code = 42
            return code
        else:
            add_failed_reason(which, video_id, opt_reason)
            print('Failure: %s  Reason: %s' % (video_id, opt_reason))
            if mode == 'only_failed':
                add_to_be_removed_from_failed(which, video_id)

    return 0


def run_parallel(mode, which, start, end, num_processes=10):
    assert mode in ['only_failed', 'og_list']
    assert which in ['train', 'valid']

    download_list = make_download_list(mode, which)
    download_list.sort()
    download_list = download_list[start:end]

    def inner_run(a_list):
        try:
            pool = Pool(processes=num_processes)
            pool.apply_async(single_run)

            outputs = pool.starmap(single_run, zip(a_list, repeat(mode), repeat(which)))

            if 429 in outputs:
                return True
            elif 42 in outputs:
                return False

        except OSError:
            print('OSError, too many open files')
            return 'oserror'

    if len(download_list) >= num_processes:

        if len(download_list) % num_processes == 0:
            steps = len(download_list) // num_processes
        else:
            steps = len(download_list) // num_processes + 1

        for i in range(steps):
            sub_list = download_list[i*num_processes:(i+1)*num_processes]
            decision = inner_run(sub_list)
            return decision
    else:
        num_processes = len(download_list)
        decision = inner_run(download_list)
        return decision



def run_parallel_and_wait():
    which = 'train'
    num_to_download = len(tools.get_failed_list(which))
    print(num_to_download)
    mode = 'only_failed'
    num_processes = 20
    start = 0

    start_date = time.strftime("%b %d %Y %H:%M:%S")

    print('====================================================================\n'
          'Downloading mode=%s, which=%s, b=%d, e=%d, parallel & wait\n'
          '====================================================================\n'
          % (mode, which, start, num_to_download))


    while num_to_download > 0:
        wait = run_parallel(mode=mode, which=which, start=start, end=num_to_download, num_processes=num_processes)

        time.sleep(5)
        clean_up_partials()
        crosscheck_lists()
        num_to_download = len(tools.get_failed_list(which))

        if wait == 'oserror':
            tools.download_progress_per_class(which)
            print('Progress plot saved')
            print('OSError encountered, stopping process')
            return

        wait_time = 900 # 15 mins
        if type(wait) is bool:
            if wait and num_to_download > 0:
                print('%s   [429 error] Waiting %d seconds before retry...' % (time.strftime("%b %d %Y %H:%M:%S"), wait_time))
                time.sleep(wait_time)

    end_date = time.strftime("%b %d %Y %H:%M:%S")
    print('=============================================================================\n'
          'Finished. Start date: %s, End date: %s\n'
          'Mode: %s, which: %s\n'
          '=============================================================================\n'
          % (start_date, end_date, mode, which))


# clean_up_partials()
# crosscheck_lists()

run_parallel_and_wait()