import os
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali import types

from config import paths as PP


def get_data(which, everything):
    ind = everything[0].index(which)
    return everything[1][ind]


def get_labels(which, everything):
    ind = everything[0].index(which)
    return everything[2][ind]


def load_jester(project_variable, seed):
    tp = np.float32
    splits = []
    all_labels = []
    all_data = []
    FRAMES = project_variable.load_num_frames  # 30


    def load(which, dp):
        label_path = os.path.join(PP.jester_location, 'labels_%s.npy' % which)
        labels = np.load(label_path)[:dp]

        data = np.zeros(shape=(dp, 3, FRAMES, 50, 75), dtype=tp)  # dp, c, d, h, w

        for i in tqdm(range(dp)):
            frames_path = os.path.join(PP.jester_data_50_75, labels[i][0])
            frames_in_folder = os.listdir(frames_path)
            frames_in_folder.sort()

            for j in range(FRAMES):
                img_path = os.path.join(PP.jester_data_50_75,
                                        str(labels[i][0]),
                                        frames_in_folder[j])

                tmp = Image.open(img_path)
                tmp = np.array(tmp)
                # tmp = np.array(tmp.convert('L'))
                tmp = tmp.transpose((2, 0, 1))
                data[i, :, j] = tmp

        labels = labels[:, 1]
        labels = [int(i) for i in labels]

        labels = np.array(labels)

        return data, labels

    def load_random(which, dp, balanced, seed):
        num_categories = 27
        total_dp = {'train': 118562, 'val': 7393, 'test': 7394}
        if dp != total_dp[which]:
            assert (dp % num_categories == 0)

        label_path = os.path.join(PP.jester_location, 'labels_%s.npy' % which)
        full_labels = np.load(label_path)
        names = full_labels[:, 0]
        labels = full_labels[:, 1]

        if balanced:
            if seed is not None:
                random.seed(seed)

            chosen = []
            for i in range(num_categories):
                indices = np.arange(total_dp[which])[labels == str(i)]

                # if seed is not None:
                #     random.seed(seed)

                num_samples_per_category = dp // num_categories

                if num_samples_per_category > len(indices):
                    chosen.extend(indices)

                    diff = num_samples_per_category - len(indices)
                    assert(len(indices) + diff == num_samples_per_category)
                    choose_indices = random.choices(list(np.arange(len(indices))), k=diff)
                else:
                    choose_indices = random.sample(list(np.arange(len(indices))), num_samples_per_category)

                chosen.extend(indices[choose_indices])
        else:
            chosen = np.arange(total_dp[which])

            if seed is not None:
                random.seed(seed)

            random.shuffle(chosen)
            chosen = chosen[:dp]

        chosen.sort()
        labels = labels[chosen]
        labels = [int(i) for i in labels]
        labels = np.array(labels)

        names = names[chosen]
        # full_labels = full_labels[chosen]

        num_points = len(labels)

        data = np.zeros(shape=(dp, 3, FRAMES, 50, 75), dtype=tp)  # dp, c, d, h, w

        for i in tqdm(range(num_points)):
            frames_path = os.path.join(PP.jester_data_50_75, names[i])
            frames_in_folder = os.listdir(frames_path)
            frames_in_folder.sort()

            for j in range(FRAMES):
                img_path = os.path.join(PP.jester_data_50_75,
                                        str(names[i]),
                                        frames_in_folder[j]
                                        )

                tmp = Image.open(img_path)
                tmp = np.array(tmp)
                tmp = tmp.transpose((2, 0, 1))
                # tmp = np.array(tmp.convert('L'))
                data[i, :, j] = tmp

        return data, labels

    if project_variable.train:
        if project_variable.randomize_training_data:
            data, some_labels = load_random('train', project_variable.data_points[0],
                                            project_variable.balance_training_data, seed)
        else:
            data, some_labels = load('train', project_variable.data_points[0])
        splits.append('train')
        all_data.append(data)
        all_labels.append(some_labels)

    if project_variable.val:
        data, some_labels = load('val', project_variable.data_points[1])
        splits.append('val')
        all_data.append(data)
        all_labels.append(some_labels)

    if project_variable.test:
        data, some_labels = load('test', project_variable.data_points[2])
        splits.append('test')
        all_data.append(data)
        all_labels.append(some_labels)

    return splits, all_data, all_labels




def load_data(project_variable, seed):
    if project_variable.dataset == 'jester':
        return load_jester(project_variable, seed)
    else:
        print('Error: dataset %s not supported' % project_variable.dataset)
        return None


# class VideoPipe(Pipeline):
#     def __init__(self, batch_size,
#                  num_threads=6,
#                  device_id=0,
#                  file_list='',
#                  shuffle=False,
#                  sequence_length=30,
#                  step=-1,
#                  stride=1,
#                  initial_fill=1024,
#                  seed=0):
#
#         super(VideoPipe, self).__init__(batch_size, num_threads, device_id, seed=seed)
#
#         self.input = ops.VideoReader(device='gpu',
#                                      file_list=file_list,
#                                      sequence_length=sequence_length,
#                                      step=step,
#                                      stride=stride,
#                                      shard_id=0,
#                                      num_shards=1,
#                                      random_shuffle=shuffle,
#                                      initial_fill=initial_fill)
#
#         self.normalize = ops.Normalize(device='gpu',
#                                        )
#
#     def define_graph(self):
#         output, labels = self.input(name="Reader")
#         return output, labels
#
#
# def create_dali_iterator(batch_size, file_list, num_workers, do_shuffle, the_seed, iterator_size, reset, device):
#
#     pipe = VideoPipe(batch_size=batch_size,
#                      file_list=file_list,
#                      shuffle=do_shuffle,
#                      initial_fill=batch_size,
#                      num_threads=num_workers,
#                      seed=the_seed,
#                      device_id=device
#                      )
#     pipe.build()
#
#     if iterator_size == 'all':
#         it_size = pipe.epoch_size("Reader")
#     else:
#         it_size = iterator_size
#
#     dali_iter = DALIGenericIterator([pipe], ['data', 'labels'], size=it_size, auto_reset=reset,
#                                     fill_last_batch=True, last_batch_padded=False)
#
#     return dali_iter
#
#
# def get_jester_iter(which, project_variable):
#     if not project_variable.xai_only_mode:
#         assert which in ['train', 'val', 'test']
#
#     if project_variable.xai_only_mode:
#         file_list = os.path.join(PP.jester_location, 'filelist_test_xai_150_224.txt')
#
#     elif project_variable.nas or project_variable.debug_mode:
#         file_list = os.path.join(PP.jester_location, 'filelist_%s_150_224_fast.txt' % which)
#     else:
#         # if project_variable.model_number in [20, 21, 23, 25]:
#         print('fetching 150 224...')
#         # default is to load from fast
#         file_list = os.path.join(PP.jester_location, 'filelist_%s_150_224_fast.txt' % which)
#         # else:
#         #     file_list = os.path.join(PP.jester_location, 'filelist_%s.txt' % which)
#
#     if which == 'val':
#         print('Loading validation iterator...')
#         the_iter = create_dali_iterator(project_variable.batch_size_val_test,
#                                         file_list, project_variable.dali_workers, False, 0,
#                                         project_variable.dali_iterator_size[1], True, project_variable.device)
#     elif which == 'test':
#         print('Loading test iterator...')
#         the_iter = create_dali_iterator(project_variable.batch_size_val_test,
#                                         file_list, project_variable.dali_workers, False, 0,
#                                         project_variable.dali_iterator_size[2], True, project_variable.device)
#     elif which == 'train':
#         print('Loading training iterator...')
#         the_iter = create_dali_iterator(project_variable.batch_size, file_list,
#                                         project_variable.dali_workers,
#                                         project_variable.randomize_training_data, 6,
#                                         project_variable.dali_iterator_size[0], True, project_variable.device)
#     else:
#         print('Loading XAI only mode iterator...')
#         the_iter = create_dali_iterator(project_variable.batch_size_val_test,
#                                         file_list, project_variable.dali_workers, False, 0,
#                                         project_variable.dali_iterator_size[1], True, project_variable.device)
#
#     return the_iter
#
#
# # ---------------------
# # jester with only 500, 200 samples per class
# class TinyJesterPipe(Pipeline):
#     def __init__(self, batch_size,
#                  num_threads=6,
#                  device_id=0,
#                  file_list='',
#                  shuffle=False,
#                  sequence_length=30,
#                  step=-1,
#                  stride=1,
#                  initial_fill=1024,
#                  seed=0):
#
#         super(TinyJesterPipe, self).__init__(batch_size, num_threads, device_id, seed=seed)
#
#         self.input = ops.VideoReader(device='gpu',
#                                      file_list=file_list,
#                                      sequence_length=sequence_length,
#                                      step=step,
#                                      stride=stride,
#                                      shard_id=0,
#                                      num_shards=1,
#                                      random_shuffle=shuffle,
#                                      initial_fill=initial_fill)
#
#         self.normalize = ops.Normalize(device='gpu',
#                                        )
#
#     def define_graph(self):
#         output, labels = self.input(name="Reader")
#         return output, labels
#
#
# def tiny_jester_iterator(batch_size, file_list, num_workers, do_shuffle, the_seed, iterator_size, reset, device):
#
#     pipe = VideoPipe(batch_size=batch_size,
#                      file_list=file_list,
#                      shuffle=do_shuffle,
#                      initial_fill=batch_size,
#                      num_threads=num_workers,
#                      seed=the_seed,
#                      device_id=device
#                      )
#     pipe.build()
#
#     if iterator_size == 'all':
#         it_size = pipe.epoch_size("Reader")
#     else:
#         it_size = iterator_size
#
#     dali_iter = DALIGenericIterator([pipe], ['data', 'labels'], size=it_size, auto_reset=reset,
#                                     fill_last_batch=True, last_batch_padded=False)
#
#     return dali_iter
#
#
# def get_tiny_jester_iter(which, project_variable):
#     # % s/\/scratch\/users\/gabras\/jester\/data_50_75_avi/\/fast\/gabras\/jester\/data_150_224_avi/g
#     if not project_variable.xai_only_mode:
#         assert which in ['train', 'val', 'test']
#
#     if which in ['train', 'test']:
#         file_list = os.path.join(PP.jester_location, 'filelist_%s_500c_150_224.txt' % which)
#     else:
#         file_list = os.path.join(PP.jester_location, 'filelist_val_200c_150_224.txt')
#
#     the_iter = None
#
#     if which == 'val':
#         print('Loading validation iterator...')
#         the_iter = tiny_jester_iterator(project_variable.batch_size_val_test,
#                                         file_list, project_variable.dali_workers, False, 0,
#                                         project_variable.dali_iterator_size[1], True, project_variable.device)
#     elif which == 'test':
#         print('Loading test iterator...')
#         the_iter = tiny_jester_iterator(project_variable.batch_size_val_test,
#                                         file_list, project_variable.dali_workers, False, 0,
#                                         project_variable.dali_iterator_size[2], True, project_variable.device)
#     elif which == 'train':
#         print('Loading training iterator...')
#         the_iter = tiny_jester_iterator(project_variable.batch_size, file_list,
#                                         project_variable.dali_workers,
#                                         project_variable.randomize_training_data, 6,
#                                         project_variable.dali_iterator_size[0], True, project_variable.device)
#
#     return the_iter

# ---------------------


# class UCF101VideoPipe(Pipeline):
#     def __init__(self, batch_size,
#                  num_threads=6,
#                  device_id=0,
#                  file_root='',
#                  shuffle=False,
#                  sequence_length=30,
#                  step=-1,
#                  stride=1,
#                  initial_fill=1024,
#                  seed=0):
#
#         super(UCF101VideoPipe, self).__init__(batch_size, num_threads, device_id, seed=seed)
#
#         self.input = ops.VideoReader(device='gpu',
#                                      file_root=file_root,
#                                      sequence_length=sequence_length,
#                                      step=step,
#                                      stride=stride,
#                                      shard_id=0,
#                                      num_shards=1,
#                                      random_shuffle=shuffle,
#                                      initial_fill=initial_fill)
#
#         self.normalize = ops.Normalize(device='gpu',
#                                        )
#
#     def define_graph(self):
#         output, labels = self.input(name="Reader")
#         return output, labels
#
#
# def ucf101_create_dali_iterator(batch_size, file_root, num_workers, do_shuffle, the_seed, iterator_size, reset, device):
#     pipe = UCF101VideoPipe(batch_size=batch_size,
#                            file_root=file_root,
#                            shuffle=do_shuffle,
#                            initial_fill=batch_size,
#                            num_threads=num_workers,
#                            seed=the_seed,
#                            device_id=device
#                            )
#     pipe.build()
#
#     if iterator_size == 'all':
#         it_size = pipe.epoch_size("Reader")
#     else:
#         it_size = iterator_size
#
#     dali_iter = DALIGenericIterator([pipe], ['data', 'labels'], size=it_size, auto_reset=reset,
#                                     fill_last_batch=True, last_batch_padded=False)
#
#     return dali_iter
#
#
# def get_ucf101_iter(which, project_variable):
#     if not project_variable.xai_only_mode:
#         assert which in ['train', 'val', 'test']
#
#     if which in ['val', 'test']:
#         print('Loading validation/test iterator...')
#         the_iter = ucf101_create_dali_iterator(project_variable.batch_size_val_test,
#                                                PP.ucf101_168_224_test, project_variable.dali_workers, False, 0,
#                                                project_variable.dali_iterator_size[1], True, project_variable.device)
#     elif which == 'train':
#         print('Loading training iterator...')
#         the_iter = ucf101_create_dali_iterator(project_variable.batch_size, PP.ucf101_168_224_train,
#                                                project_variable.dali_workers,
#                                                project_variable.randomize_training_data, 6,
#                                                project_variable.dali_iterator_size[0], True, project_variable.device)
#     else:
#         print('Loading XAI only mode iterator...')
#         the_iter = ucf101_create_dali_iterator(project_variable.batch_size_val_test,
#                                                PP.ucf101_168_224_xai, project_variable.dali_workers, False, 0,
#                                                project_variable.dali_iterator_size[1], True, project_variable.device)
#
#     return the_iter
#
#
# class Kinetics400VideoPipe(Pipeline):
#     def __init__(self, batch_size,
#                  num_threads=6,
#                  device_id=0,
#                  file_root='',
#                  shuffle=False,
#                  sequence_length=30,
#                  step=-1,
#                  stride=1,
#                  initial_fill=1024,
#                  seed=0):
#
#         super(Kinetics400VideoPipe, self).__init__(batch_size, num_threads, device_id, seed=seed)
#
#         self.input = ops.VideoReader(device='gpu',
#                                      file_root=file_root,
#                                      sequence_length=sequence_length,
#                                      step=step,
#                                      stride=stride,
#                                      shard_id=0,
#                                      num_shards=1,
#                                      random_shuffle=shuffle,
#                                      initial_fill=initial_fill)
#
#         self.normalize = ops.Normalize(device='gpu',
#                                        )
#
#     def define_graph(self):
#         output, labels = self.input(name="Reader")
#         return output, labels
#
#
# def kinetics400_create_dali_iterator(batch_size, file_root, num_workers, do_shuffle, the_seed, iterator_size, reset, device):
#     pipe = Kinetics400VideoPipe(batch_size=batch_size,
#                                 file_root=file_root,
#                                 shuffle=do_shuffle,
#                                 initial_fill=batch_size,
#                                 num_threads=num_workers,
#                                 seed=the_seed,
#                                 device_id=device
#                                 )
#     pipe.build()
#
#     if iterator_size == 'all':
#         it_size = pipe.epoch_size("Reader")
#     else:
#         it_size = iterator_size
#
#     dali_iter = DALIGenericIterator([pipe], ['data', 'labels'], size=it_size, auto_reset=reset,
#                                     fill_last_batch=True, last_batch_padded=False)
#
#     return dali_iter
#
#
# def get_kinetics400_iter(which, project_variable):
#     if not project_variable.xai_only_mode:
#         assert which in ['train', 'val', 'test']
#
#     if which in ['val']:
#         print('Loading validation/test iterator...')
#         the_iter = kinetics400_create_dali_iterator(project_variable.batch_size_val_test,
#                                                     PP.kinetics400_val,
#                                                     project_variable.dali_workers, False, 0,
#                                                     project_variable.dali_iterator_size[1], True,
#                                                     project_variable.device)
#     elif which in ['test']:
#         print('Loading validation/test iterator...')
#         the_iter = kinetics400_create_dali_iterator(project_variable.batch_size_val_test,
#                                                     PP.kinetics400_test,
#                                                     project_variable.dali_workers, False, 0,
#                                                     project_variable.dali_iterator_size[1], True,
#                                                     project_variable.device)
#     elif which == 'train':
#         print('Loading training iterator...')
#         the_iter = kinetics400_create_dali_iterator(project_variable.batch_size,
#                                                     PP.kinetics400_train,
#                                                     project_variable.dali_workers,
#                                                     project_variable.randomize_training_data, 6,
#                                                     project_variable.dali_iterator_size[0], True,
#                                                     project_variable.device)
#     else:
#         print('Error: which %s is not recognized' % which)
#         return
#
#     return the_iter




# ---------------------

# Universal pipeline for datasets using file_list=file_list (separate txt file with "videopath,class")
class VideoPipeFileList(Pipeline):
    def __init__(self, batch_size,
                 num_threads=6,
                 device_id=0,
                 file_list='',
                 shuffle=False,
                 sequence_length=30,
                 step=-1,
                 stride=1,
                 initial_fill=1024,
                 seed=0):

        super(VideoPipeFileList, self).__init__(batch_size, num_threads, device_id, seed=seed)

        self.input = ops.VideoReader(device='gpu',
                                     file_list=file_list,
                                     sequence_length=sequence_length,
                                     step=step,
                                     stride=stride,
                                     shard_id=0,
                                     num_shards=1,
                                     random_shuffle=shuffle,
                                     initial_fill=initial_fill)

        self.normalize = ops.Normalize(device='gpu')

    def define_graph(self):
        output, labels = self.input(name="Reader")
        return output, labels


def file_list_dali_iterator(batch_size, file_list, num_workers, do_shuffle, the_seed, iterator_size, reset, device, 
                            num_frames):
    pipe = VideoPipeFileList(batch_size=batch_size,
                             file_list=file_list,
                             shuffle=do_shuffle,
                             initial_fill=batch_size,
                             num_threads=num_workers,
                             seed=the_seed,
                             device_id=device,
                             sequence_length=num_frames)
    pipe.build()

    if iterator_size == 'all':
        it_size = pipe.epoch_size("Reader")
    else:
        it_size = iterator_size

    dali_iter = DALIGenericIterator([pipe], ['data', 'labels'], size=it_size, auto_reset=reset,
                                    fill_last_batch=True, last_batch_padded=False)

    return dali_iter


def get_file_list_iterator(which, project_variable, path_train, path_val, path_test, path_xai):
    assert which in ['train', 'val', 'test', 'xai']

    num_workers = project_variable.dali_workers
    reset = True
    device = project_variable.device
    num_frames = project_variable.load_num_frames

    if which in ['val']:
        print('Loading iterator for validation...')
        batch_size = project_variable.batch_size_val_test
        file_list = path_val
        do_shuffle = False
        the_seed = 0
        iterator_size = project_variable.dali_iterator_size[1]

    elif which in ['test']:
        print('Loading iterator for test...')
        batch_size = project_variable.batch_size_val_test
        file_list = path_test
        do_shuffle = False
        the_seed = 0
        iterator_size = project_variable.dali_iterator_size[2]

    elif which == 'train':
        print('Loading iterator for train...')
        batch_size = project_variable.batch_size
        file_list = path_train
        do_shuffle = project_variable.randomize_training_data
        the_seed = 6
        iterator_size = project_variable.dali_iterator_size[0]

    elif which == 'xai':
        print('Loading iterator for xai...')
        batch_size = project_variable.batch_size_val_test
        file_list = path_xai
        do_shuffle = False
        the_seed = 0
        iterator_size = project_variable.dali_iterator_size[1]

    the_iter = file_list_dali_iterator(batch_size, file_list, num_workers, do_shuffle, the_seed, iterator_size, reset,
                                       device, num_frames)

    return the_iter
# ---------------------

# Universal pipeline for datasets using file_root=file_root (samples are in separate class folders)
class VideoPipeFileRoot(Pipeline):
    def __init__(self, batch_size,
                 num_threads=6,
                 device_id=0,
                 file_root='',
                 shuffle=False,
                 sequence_length=30,
                 step=-1,
                 stride=1,
                 initial_fill=1024,
                 seed=0,
                 channels=3):

        super(VideoPipeFileRoot, self).__init__(batch_size, num_threads, device_id, seed=seed)

        self.input = ops.VideoReader(device='gpu',
                                     file_root=file_root,
                                     sequence_length=sequence_length,
                                     step=step,
                                     stride=stride,
                                     shard_id=0,
                                     num_shards=1,
                                     random_shuffle=shuffle,
                                     initial_fill=initial_fill,
                                     channels=channels)

        self.normalize = ops.Normalize(device='gpu')

    def define_graph(self):
        output, labels = self.input(name="Reader")
        return output, labels


def file_root_dali_iterator(batch_size, file_root, num_workers, do_shuffle, the_seed, iterator_size, reset, device,
                            num_frames, channels):
    pipe = VideoPipeFileRoot(batch_size=batch_size,
                             file_root=file_root,
                             shuffle=do_shuffle,
                             initial_fill=batch_size,
                             num_threads=num_workers,
                             seed=the_seed,
                             device_id=device,
                             sequence_length=num_frames,
                             channels=channels)
    pipe.build()

    if iterator_size == 'all':
        it_size = pipe.epoch_size("Reader")
    else:
        it_size = iterator_size

    dali_iter = DALIGenericIterator([pipe], ['data', 'labels'], size=it_size, auto_reset=reset,
                                    fill_last_batch=True, last_batch_padded=False)

    return dali_iter


def get_file_root_iterator(which, project_variable, path_train, path_val, path_test, path_xai, chan):
    assert which in ['train', 'val', 'test', 'xai']

    num_workers = project_variable.dali_workers
    reset = True
    device = project_variable.device
    num_frames = project_variable.load_num_frames
    channels = chan

    if which in ['val']:
        print('Loading iterator for validation...')
        batch_size = project_variable.batch_size_val_test
        file_root = path_val
        do_shuffle = False
        the_seed = 0
        iterator_size = project_variable.dali_iterator_size[1]

    elif which in ['test']:
        print('Loading iterator for test...')
        batch_size = project_variable.batch_size_val_test
        file_root = path_test
        do_shuffle = False
        the_seed = 0
        iterator_size = project_variable.dali_iterator_size[1]

    elif which == 'train':
        print('Loading iterator for train...')
        batch_size = project_variable.batch_size
        file_root = path_train
        do_shuffle = project_variable.randomize_training_data
        the_seed = 6
        iterator_size = project_variable.dali_iterator_size[0]

    elif which == 'xai':
        print('Loading iterator for xai...')
        batch_size = project_variable.batch_size_val_test
        file_root = path_xai
        do_shuffle = False
        the_seed = 0
        iterator_size = project_variable.dali_iterator_size[1]


    the_iter = file_root_dali_iterator(batch_size, file_root, num_workers, do_shuffle, the_seed, iterator_size, reset,
                                       device, num_frames, channels)
    return the_iter



class ImagePipeFileRoot(Pipeline):
    def __init__(self, batch_size,
                 num_threads=6,
                 device_id=0,
                 file_root='',
                 shuffle=False,
                 step=-1,
                 stride=1,
                 initial_fill=1024,
                 seed=0):

        super(ImagePipeFileRoot, self).__init__(batch_size, num_threads, device_id, seed=seed)

        self.input = ops.FileReader(file_root=file_root, random_shuffle=shuffle, initial_fill=initial_fill, shard_id=0,
                                    num_shards=1)
        self.decode = ops.ImageDecoder(device='mixed', output_type=types.GRAY, hw_decoder_load=0.65)


    def define_graph(self):
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        return images, labels


def image_file_root_iterator(batch_size, file_root, num_workers, do_shuffle, the_seed, iterator_size, reset, device):
    pipe = ImagePipeFileRoot(batch_size=batch_size,
                             file_root=file_root,
                             shuffle=do_shuffle,
                             initial_fill=batch_size,
                             num_threads=num_workers,
                             seed=the_seed,
                             device_id=device)
    pipe.build()

    if iterator_size == 'all':
        it_size = pipe.epoch_size("Reader")
    else:
        it_size = iterator_size

    dali_iter = DALIGenericIterator([pipe], ['data', 'labels'], size=it_size, auto_reset=reset,
                                    fill_last_batch=True, last_batch_padded=False)

    return dali_iter


def get_image_file_root_iterator(which, project_variable, path_train, path_val, path_test, path_xai):
    assert which in ['train', 'val', 'test', 'xai']

    num_workers = project_variable.dali_workers
    reset = True
    device = project_variable.device
    num_frames = project_variable.load_num_frames

    if which in ['val']:
        print('Loading iterator for validation...')
        batch_size = project_variable.batch_size_val_test
        file_root = path_val
        do_shuffle = False
        the_seed = 0
        iterator_size = project_variable.dali_iterator_size[1]

    elif which in ['test']:
        print('Loading iterator for test...')
        batch_size = project_variable.batch_size_val_test
        file_root = path_test
        do_shuffle = False
        the_seed = 0
        iterator_size = project_variable.dali_iterator_size[1]

    elif which == 'train':
        print('Loading iterator for train...')
        batch_size = project_variable.batch_size
        file_root = path_train
        do_shuffle = project_variable.randomize_training_data
        the_seed = 6
        iterator_size = project_variable.dali_iterator_size[0]

    elif which == 'xai':
        print('Loading iterator for xai...')
        batch_size = project_variable.batch_size_val_test
        file_root = path_xai
        do_shuffle = False
        the_seed = 0
        iterator_size = project_variable.dali_iterator_size[1]

    the_iter = image_file_root_iterator(batch_size, file_root, num_workers, do_shuffle, the_seed, iterator_size, reset,
                                        device)
    return the_iter


def get_iterator(which, project_variable):
    # get_file_root_iterator(which, project_variable, path_train, path_val, path_test, path_xai)
    
    iterator = None
    
    if project_variable.dataset == 'jester':
        p_train = os.path.join(PP.jester_location, 'filelist_train_150_224_fast.txt')
        p_val = os.path.join(PP.jester_location, 'filelist_val_150_224_fast.txt')
        p_test = os.path.join(PP.jester_location, 'filelist_test_150_224_fast.txt')
        p_xai = os.path.join(PP.jester_location, 'filelist_test_xai_150_224.txt')

        iterator = get_file_list_iterator(which, project_variable, p_train, p_val, p_test, p_xai)

    elif project_variable.dataset == 'tiny_jester':
        p_train = os.path.join(PP.jester_location, 'filelist_train_500c_150_224.txt')
        p_val = os.path.join(PP.jester_location, 'filelist_val_200c_150_224.txt')
        p_test = os.path.join(PP.jester_location, 'filelist_test_500c_150_224.txt')
        p_xai = None
        iterator = get_file_list_iterator(which, project_variable, p_train, p_val, p_test, p_xai)

    elif project_variable.dataset == 'ucf101':
        channels = 3
        iterator = get_file_root_iterator(which, project_variable, PP.ucf101_168_224_train, PP.ucf101_168_224_test,
                                          PP.ucf101_168_224_test, PP.ucf101_168_224_xai, channels)

    elif project_variable.dataset == 'kinetics400':
        iterator = get_file_root_iterator(which, project_variable, PP.kinetics400_train, PP.kinetics400_val,
                                          PP.kinetics400_test, None, 3)

    elif project_variable.dataset == 'dots_avi':
        p_train = os.path.join(PP.dots_dataset_avi, 'train')
        p_val = os.path.join(PP.dots_dataset_avi, 'val')
        p_test = os.path.join(PP.dots_dataset_avi, 'test')
        p_xai = None
        channels = 1
        iterator = get_file_root_iterator(which, project_variable, p_train, p_val, p_test, p_xai, channels)

    elif project_variable.dataset == 'dots_frames':
        p_train = os.path.join(PP.dots_dataset_frames, 'train')
        p_val = os.path.join(PP.dots_dataset_frames, 'val')
        p_test = os.path.join(PP.dots_dataset_frames, 'test')
        p_xai = None

        iterator = get_image_file_root_iterator(which, project_variable, p_train, p_val, p_test, p_xai)

    elif project_variable.dataset == 'kinetics400_metaclass':
        p_train = PP.kinetics400_train_meta
        p_val = PP.kinetics400_val_meta
        p_test = None
        p_xai = None
        iterator = get_file_list_iterator(which, project_variable, p_train, p_val, p_test, p_xai)

    return iterator
