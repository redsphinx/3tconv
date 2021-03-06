import time

from config.base_config import ProjectVariable
from utilities import utils
from running import main_file


def set_init_1():
    project_variable.end_epoch = 100
    project_variable.dataset = 'jester'

    # total_dp = {'train': 118562, 'val': 7393, 'test': 7394}
    project_variable.num_in_channels = 3
    project_variable.data_points = [2 * 27,  1 * 27, 0 * 27]
    project_variable.label_size = 27
    project_variable.batch_size = 5 * 27
    project_variable.load_num_frames = 30
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


def e26_conv3D_jester():
    set_init_1()
    project_variable.model_number = 21
    project_variable.experiment_number = 26
    project_variable.sheet_number = 22
    project_variable.device = 0
    project_variable.end_epoch = 100
    project_variable.repeat_experiments = 1
    project_variable.batch_size = 32
    project_variable.batch_size_val_test = 32

    project_variable.load_model = True
    project_variable.load_from_fast = True

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.stop_at_collapse = True
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.000005
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [0]

    main_file.run(project_variable)


def e28_conv3D_jester():
    set_init_1()
    project_variable.model_number = 25
    project_variable.experiment_number = 28
    project_variable.sheet_number = 22
    project_variable.device = 0
    project_variable.end_epoch = 100
    project_variable.repeat_experiments = 1
    project_variable.batch_size = 19
    project_variable.batch_size_val_test = 19

    project_variable.load_model = True
    project_variable.load_from_fast = True

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.stop_at_collapse = True
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.000005
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [0]

    main_file.run(project_variable)


def e29_conv3T_jester():
    set_init_1()
    project_variable.model_number = 20
    project_variable.experiment_number = 29
    project_variable.sheet_number = 22
    project_variable.device = 1
    project_variable.end_epoch = 100
    project_variable.repeat_experiments = 1
    project_variable.batch_size = 32
    project_variable.batch_size_val_test = 32

    project_variable.load_model = True
    project_variable.load_from_fast = True

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.stop_at_collapse = True
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.00005
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [0]

    go = False
    while not go:
        gpu_available = get_gpu_memory_map()
        if gpu_available[project_variable.device] < 100:
            go = True
        else:
            print('waiting for gpu...')
            time.sleep(10)

    main_file.run(project_variable)


def e30_conv3T_jester():
    set_init_1()
    project_variable.model_number = 23
    project_variable.experiment_number = 30
    project_variable.sheet_number = 22
    project_variable.device = 1
    project_variable.end_epoch = 100
    project_variable.repeat_experiments = 1
    project_variable.batch_size = 20
    project_variable.batch_size_val_test = 20

    project_variable.load_model = True
    project_variable.load_from_fast = True

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.stop_at_collapse = True
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.00005
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [0]

    main_file.run(project_variable)


def e31_conv3T_jester():
    set_init_1()
    project_variable.model_number = 20
    project_variable.experiment_number = 31
    project_variable.sheet_number = 22
    project_variable.device = 2
    project_variable.end_epoch = 100
    project_variable.repeat_experiments = 1
    project_variable.batch_size = 32
    project_variable.batch_size_val_test = 32

    project_variable.load_model = [29, 20, 13, 0]  # exp, model, epoch, run
    project_variable.load_from_fast = True

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.stop_at_collapse = True
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.00005
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [0]

    go = False
    while not go:
        gpu_available = get_gpu_memory_map()
        if gpu_available[project_variable.device] < 100:
            go = True
        else:
            print('waiting for gpu...')
            time.sleep(10)

    main_file.run(project_variable)


def e32_conv3T_jester():
    set_init_1()
    project_variable.model_number = 20 # RN18 3T
    project_variable.experiment_number = 32
    project_variable.sheet_number = 22
    project_variable.device = 1
    project_variable.end_epoch = 100
    project_variable.repeat_experiments = 1
    project_variable.batch_size = 32
    project_variable.batch_size_val_test = 32

    project_variable.load_model = None
    project_variable.load_from_fast = True

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.stop_at_collapse = True
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.00005
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [0]

    wait_for_gpu(wait=False, device_num=project_variable.device)
    main_file.run(project_variable)


def e33_conv3T_jester():
    set_init_1()
    project_variable.model_number = 23 # GN 3T
    project_variable.experiment_number = 33
    project_variable.sheet_number = 22
    project_variable.device = 1
    project_variable.end_epoch = 100
    project_variable.repeat_experiments = 1
    project_variable.batch_size = 20
    project_variable.batch_size_val_test = 20

    project_variable.load_model = None
    project_variable.load_from_fast = True

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.stop_at_collapse = True
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.00005
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [0]

    wait_for_gpu(wait=True, device_num=project_variable.device)
    main_file.run(project_variable)


def e36_conv3T_jester():
    set_init_1()
    project_variable.model_number = 20 # RN18 3T
    project_variable.experiment_number = 36
    project_variable.sheet_number = 22
    project_variable.device = 2
    project_variable.end_epoch = 100
    project_variable.repeat_experiments = 1
    project_variable.batch_size = 32
    project_variable.batch_size_val_test = 32

    project_variable.load_model = [32, 20, 13, 0]
    project_variable.load_from_fast = True

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.stop_at_collapse = True
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.00005
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [0]

    wait_for_gpu(wait=True, device_num=project_variable.device)
    main_file.run(project_variable)



def get_sizes():
    set_init_1()
    project_variable.nin = False
    project_variable.model_number = 20 # RN18 3T
    project_variable.experiment_number = 37234234234234234234
    project_variable.sheet_number = 22
    project_variable.device = 1
    project_variable.end_epoch = 1
    project_variable.repeat_experiments = 1
    project_variable.batch_size = 1
    project_variable.batch_size_val_test = 1

    project_variable.load_model = True
    project_variable.load_from_fast = True

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.stop_at_collapse = True
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.00005
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [0]

    # wait_for_gpu(wait=True, device_num=project_variable.device)
    main_file.run(project_variable)

# 1:  torch.Size([1, 3, 30, 150, 224])
# 2:  torch.Size([1, 64, 8, 38, 56])
# 3:  torch.Size([1, 64, 8, 38, 56])
# 4:  torch.Size([1, 64, 8, 38, 56])
# 5:  torch.Size([1, 64, 8, 38, 56])
# 7:  torch.Size([1, 64, 8, 38, 56])
# 8:  torch.Size([1, 128, 4, 19, 28])
# 9:  torch.Size([1, 128, 4, 19, 28])
# 10:  torch.Size([1, 128, 4, 19, 28])
# 12:  torch.Size([1, 128, 4, 19, 28])
# 13:  torch.Size([1, 256, 2, 10, 14])
# 14:  torch.Size([1, 256, 2, 10, 14])
# 15:  torch.Size([1, 256, 2, 10, 14])
# 17:  torch.Size([1, 256, 2, 10, 14])
# 18:  torch.Size([1, 512, 1, 5, 7])
# 19:  torch.Size([1, 512, 1, 5, 7])
# 20:  torch.Size([1, 512, 1, 5, 7])


def e37_conv3T_jester():
    set_init_1()
    project_variable.nin = True
    project_variable.train_nin_mode = 'nin_only'

    project_variable.model_number = 50 # RN18 3T
    project_variable.experiment_number = 37
    project_variable.sheet_number = 22

    project_variable.device = 1
    project_variable.end_epoch = 5
    project_variable.repeat_experiments = 1
    project_variable.batch_size = 1
    project_variable.batch_size_val_test = 1

    project_variable.load_model = [31, 20, 8, 0]
    project_variable.load_from_fast = True

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.stop_at_collapse = True
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.00005
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [0]

    # wait_for_gpu(wait=True, device_num=project_variable.device)
    main_file.run(project_variable)


def e38_conv3T_jester():
    set_init_1()
    project_variable.nin = False
    # project_variable.train_nin_mode = 'nin_only'

    project_variable.model_number = 51 # convnet3T
    project_variable.experiment_number = 38
    project_variable.sheet_number = 22

    project_variable.device = 0
    project_variable.end_epoch = 10
    project_variable.repeat_experiments = 1
    project_variable.batch_size = 20
    project_variable.batch_size_val_test = 20

    project_variable.load_model = None
    project_variable.load_from_fast = True

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.stop_at_collapse = True
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.00005
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [0]

    # wait_for_gpu(wait=True, device_num=project_variable.device)
    main_file.run(project_variable)


def e39_conv3T_jester():
    set_init_1()
    project_variable.nin = True
    project_variable.train_nin_mode = 'joint'

    project_variable.model_number = 52 # TACoNet
    project_variable.experiment_number = 39
    project_variable.sheet_number = 22

    project_variable.device = 1
    project_variable.end_epoch = 10
    project_variable.repeat_experiments = 1
    project_variable.batch_size = 20
    project_variable.batch_size_val_test = 20

    project_variable.load_model = None
    project_variable.load_from_fast = True

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.stop_at_collapse = True
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.00005
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [0]

    # wait_for_gpu(wait=True, device_num=project_variable.device)
    main_file.run(project_variable)


def e40_conv3T_jester():
    set_init_1()
    project_variable.dataset = 'tiny_jester'
    project_variable.nin = False
    # project_variable.train_nin_mode = 'nin_only'

    project_variable.model_number = 51 # convnet3T
    project_variable.experiment_number = 40
    project_variable.sheet_number = 22

    project_variable.device = 2
    project_variable.end_epoch = 100
    project_variable.repeat_experiments = 10
    project_variable.batch_size = 20
    project_variable.batch_size_val_test = 20

    project_variable.load_model = None
    project_variable.load_from_fast = True

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.stop_at_collapse = True
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.00005
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [0]

    wait_for_gpu(wait=True, device_num=project_variable.device, threshold=2000)
    main_file.run(project_variable)


def e41_conv3T_jester():
    set_init_1()
    project_variable.dataset = 'tiny_jester'
    project_variable.nin = True
    project_variable.train_nin_mode = 'joint'

    project_variable.model_number = 52 # TACoNet
    project_variable.experiment_number = 41
    project_variable.sheet_number = 22

    project_variable.device = 0
    project_variable.end_epoch = 100
    project_variable.repeat_experiments = 10
    project_variable.batch_size = 20
    project_variable.batch_size_val_test = 20

    project_variable.load_model = None
    project_variable.load_from_fast = True

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.stop_at_collapse = True
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.00005
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [0]

    # wait_for_gpu(wait=True, device_num=project_variable.device)
    main_file.run(project_variable)


def e42_conv3T_jester():
    set_init_1()
    project_variable.dataset = 'tiny_jester'
    project_variable.nin = False
    # project_variable.train_nin_mode = 'joint'

    project_variable.model_number = 53 # alexnet3t
    project_variable.experiment_number = 42
    project_variable.sheet_number = 22

    project_variable.device = 1
    project_variable.end_epoch = 100
    project_variable.repeat_experiments = 10
    project_variable.batch_size = 20
    project_variable.batch_size_val_test = 20

    project_variable.load_model = True
    project_variable.load_from_fast = True

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.stop_at_collapse = True
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.00005
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [0]

    # wait_for_gpu(wait=True, device_num=project_variable.device)
    main_file.run(project_variable)


def e43_conv3T_jester():
    set_init_1()
    project_variable.dataset = 'tiny_jester'
    project_variable.nin = True
    project_variable.train_nin_mode = 'joint'

    project_variable.model_number = 54 # alexnet-taco
    project_variable.experiment_number = 43
    project_variable.sheet_number = 22

    project_variable.device = 0
    project_variable.end_epoch = 100
    project_variable.repeat_experiments = 3
    project_variable.batch_size = 20
    project_variable.batch_size_val_test = 20

    project_variable.load_model = True
    project_variable.load_from_fast = True

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.stop_at_collapse = True
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.00005
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [0]

    # wait_for_gpu(wait=True, device_num=project_variable.device)
    main_file.run(project_variable)


def e44_conv3T_jester():
    set_init_1()
    project_variable.dataset = 'tiny_jester'
    project_variable.nin = True
    project_variable.train_nin_mode = 'joint'

    project_variable.model_number = 54 # alexnet-taco
    project_variable.experiment_number = 44
    project_variable.sheet_number = 22

    project_variable.device = 0
    project_variable.end_epoch = 100
    project_variable.repeat_experiments = 3
    project_variable.batch_size = 20
    project_variable.batch_size_val_test = 20

    project_variable.load_model = True
    project_variable.load_from_fast = True

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.stop_at_collapse = True
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.000005
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [0]

    # utils.wait_for_gpu(wait=True, device_num=project_variable.device)
    main_file.run(project_variable)


def e45_conv3T_jester():
    set_init_1()
    project_variable.dataset = 'tiny_jester'
    project_variable.nin = True
    project_variable.train_nin_mode = 'joint'

    project_variable.model_number = 54 # alexnet-taco
    project_variable.experiment_number = 45
    project_variable.sheet_number = 22

    project_variable.device = 0
    project_variable.end_epoch = 100
    project_variable.repeat_experiments = 3
    project_variable.batch_size = 20
    project_variable.batch_size_val_test = 20

    project_variable.load_model = True
    project_variable.load_from_fast = True

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.stop_at_collapse = True
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.00001
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [0]

    # utils.wait_for_gpu(wait=True, device_num=project_variable.device)
    main_file.run(project_variable)

# BUG: experiment number 50 is taken already, because of other project with R(2+)D


# lovelace
def e100_conv3D_jester():
    set_init_1()
    project_variable.model_number = 25 # 3Dgooglenet from scratch
    project_variable.experiment_number = 100
    project_variable.sheet_number = 22
    project_variable.device = 1
    project_variable.end_epoch = 100
    project_variable.repeat_experiments = 1
    project_variable.batch_size = 20
    project_variable.batch_size_val_test = 20

    project_variable.load_model = False
    project_variable.load_from_fast = True

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.stop_at_collapse = True
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.000005
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [0]

    main_file.run(project_variable)


# godel 99153.pts-22.godel
def e101_conv3D_jester():
    set_init_1()
    project_variable.model_number = 21 # 3Dresnet18 from scratch
    project_variable.experiment_number = 101
    project_variable.sheet_number = 22
    project_variable.device = 2
    project_variable.end_epoch = 100
    project_variable.repeat_experiments = 1
    project_variable.batch_size = 32
    project_variable.batch_size_val_test = 32

    project_variable.load_model = False
    project_variable.load_from_fast = True

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.stop_at_collapse = True
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.000005
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [0]

    main_file.run(project_variable)


# project_variable = ProjectVariable(debug_mode=True)
project_variable = ProjectVariable(debug_mode=False)


e100_conv3D_jester()  # godel
# e101_conv3D_jester()  # godel