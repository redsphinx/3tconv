from datetime import datetime
import subprocess
import time

from config.base_config import ProjectVariable
from running import main_file


def get_gpu_memory_map():
    # from: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/3
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def wait_for_gpu(wait, device_num=None, threshold=100):

    if wait:
        go = False
        while not go:
            gpu_available = get_gpu_memory_map()
            if gpu_available[device_num] < threshold:
                go = True
            else:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print('%s Waiting for gpu %d...' % (current_time, device_num))
                time.sleep(10)
    else:
        return


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


def e37_conv3T_jester():
    set_init_1()
    project_variable.model_number = 20 # RN18 3T
    project_variable.experiment_number = 37
    project_variable.sheet_number = 22
    project_variable.device = 1
    project_variable.end_epoch = 5
    project_variable.repeat_experiments = 1
    project_variable.batch_size = 1
    project_variable.batch_size_val_test = 1

    project_variable.load_model = [36, 20, 13, 0]
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

# project_variable = ProjectVariable(debug_mode=False)
project_variable = ProjectVariable(debug_mode=True)

e37_conv3T_jester()