from config.base_config import ProjectVariable
from running import main_file
from utilities import utils


def set_init_1():
    project_variable.end_epoch = 100
    project_variable.dataset = 'ucf101'

    # total_dp = {'train': 9537, 'val/test': 3783}
    project_variable.num_in_channels = 3
    project_variable.data_points = [2 * 27,  1 * 27, 0 * 27]
    project_variable.label_size = 101
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


def e1000_3D_ucf101():
    set_init_1()
    project_variable.model_number = 21
    project_variable.experiment_number = 1000
    project_variable.sheet_number = 23
    project_variable.device = 2
    project_variable.end_epoch = 100
    project_variable.batch_size = 30
    project_variable.batch_size_val_test = 30

    project_variable.load_model = True  # exp, model, epoch, run
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

    main_file.run(project_variable)


def e1001_3T_ucf101():
    set_init_1()
    project_variable.model_number = 20
    project_variable.experiment_number = 1001
    project_variable.sheet_number = 23
    project_variable.device = 0
    project_variable.end_epoch = 100
    project_variable.batch_size = 30
    project_variable.batch_size_val_test = 30

    project_variable.load_model = True  # exp, model, epoch, run
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

    main_file.run(project_variable)


def e1002_3D_ucf101():
    set_init_1()
    project_variable.model_number = 25
    project_variable.experiment_number = 1002
    project_variable.sheet_number = 23
    project_variable.device = 1
    project_variable.end_epoch = 100
    project_variable.batch_size = 20
    project_variable.batch_size_val_test = 20

    project_variable.load_model = True  # exp, model, epoch, run
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

    main_file.run(project_variable)


def e1003_3T_ucf101():
    set_init_1()
    project_variable.model_number = 23
    project_variable.experiment_number = 1003
    project_variable.sheet_number = 23
    project_variable.device = 0
    project_variable.end_epoch = 100
    project_variable.batch_size = 19
    project_variable.batch_size_val_test = 19

    project_variable.load_model = True  # exp, model, epoch, run
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

    main_file.run(project_variable)


def e1005_3T_ucf101():
    set_init_1()
    project_variable.model_number = 23
    project_variable.experiment_number = 1005
    project_variable.sheet_number = 23
    project_variable.device = 2
    project_variable.end_epoch = 100
    project_variable.batch_size = 18
    project_variable.batch_size_val_test = 18

    project_variable.load_model = None  # exp, model, epoch, run
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

    main_file.run(project_variable)


def e1008_3T_ucf101():
    set_init_1()
    project_variable.model_number = 20
    project_variable.experiment_number = 1008
    project_variable.sheet_number = 23
    project_variable.device = 0
    project_variable.end_epoch = 100
    project_variable.batch_size = 16
    project_variable.batch_size_val_test = 16

    # load from 23, model 24 might be corrupted
    project_variable.load_model = [1004, 20, 23, 0]  # exp, model, epoch, run
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

    main_file.run(project_variable)


def e1009_3T_ucf101():
    set_init_1()
    project_variable.model_number = 20
    project_variable.experiment_number = 1009
    project_variable.sheet_number = 23
    project_variable.device = 1
    project_variable.end_epoch = 5
    project_variable.batch_size = 1
    project_variable.batch_size_val_test = 1

    # load from 23, model 24 might be corrupted
    project_variable.load_model = [1008, 20, 11, 0]  # exp, model, epoch, run
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

    main_file.run(project_variable)


def e1010_3T_ucf101():
    set_init_1()
    project_variable.nin = False

    project_variable.model_number = 51 # convnet3T
    project_variable.experiment_number = 38
    project_variable.sheet_number = 23

    project_variable.device = 1
    project_variable.end_epoch = 1
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

    # utils.wait_for_gpu(wait=True, device_num=project_variable.device)
    main_file.run(project_variable)


def e1011_3T_ucf101():
    set_init_1()
    project_variable.nin = True
    project_variable.train_nin_mode = 'joint'

    project_variable.model_number = 52 # TACoNet
    project_variable.experiment_number = 1011
    project_variable.sheet_number = 22

    project_variable.device = 1
    project_variable.end_epoch = 1
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

    main_file.run(project_variable)


def e1012_3T_ucf101():
    set_init_1()
    project_variable.model_number = 23
    project_variable.experiment_number = 1012
    project_variable.sheet_number = 23
    project_variable.device = 1
    project_variable.end_epoch = 100
    project_variable.batch_size = 15
    project_variable.batch_size_val_test = 15

    project_variable.load_model = [1, 23, 4, 0]  # exp, model, epoch, run
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

    utils.wait_for_gpu(wait=True, device_num=project_variable.device, threshold=9700)
    main_file.run(project_variable)


def e1013_3T_ucf101():
    set_init_1()
    project_variable.model_number = 23
    project_variable.experiment_number = 1013
    project_variable.sheet_number = 23
    # project_variable.device = 1  # original
    project_variable.device = 0  # for debugging
    project_variable.end_epoch = 100
    project_variable.batch_size = 15
    project_variable.batch_size_val_test = 15

    project_variable.load_model = [1, 23, 4, 0]  # exp, model, epoch, run
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

    # utils.wait_for_gpu(wait=True, device_num=project_variable.device, threshold=9700)
    main_file.run(project_variable)


def e1014_3T_ucf101():
    set_init_1()
    project_variable.model_number = 23
    project_variable.experiment_number = 1014
    project_variable.sheet_number = 23
    project_variable.device = 1
    project_variable.end_epoch = 100
    project_variable.batch_size = 15
    project_variable.batch_size_val_test = 15

    project_variable.load_model = [1, 23, 4, 0]  # exp, model, epoch, run
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

    utils.wait_for_gpu(wait=True, device_num=project_variable.device, threshold=9700)
    main_file.run(project_variable)


def e1015_3D_ucf101():
    set_init_1()
    project_variable.model_number = 25 # 3Dgooglenet from scratch
    project_variable.experiment_number = 1015
    project_variable.sheet_number = 23
    project_variable.device = 0
    project_variable.end_epoch = 100
    project_variable.batch_size = 30
    project_variable.batch_size_val_test = 30

    project_variable.load_model = False  # exp, model, epoch, run
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

    main_file.run(project_variable)


def e1016_3D_ucf101():
    set_init_1()
    project_variable.model_number = 21 # 3Dresnet18 from scratch
    project_variable.experiment_number = 1016
    project_variable.sheet_number = 23
    project_variable.device = 1
    project_variable.end_epoch = 100
    project_variable.batch_size = 25
    project_variable.batch_size_val_test = 25

    project_variable.load_model = False  # exp, model, epoch, run
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

    main_file.run(project_variable)


def e1017_3D_ucf101():
    set_init_1()
    project_variable.model_number = 25 # 3Dgooglenet from scratch
    project_variable.experiment_number = 1017
    project_variable.sheet_number = 23
    project_variable.device = 0
    project_variable.end_epoch = 100
    project_variable.batch_size = 20
    project_variable.batch_size_val_test = 20

    project_variable.load_model = False  # exp, model, epoch, run
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

    main_file.run(project_variable)


def e1018_3D_ucf101():
    set_init_1()
    project_variable.model_number = 25 # 3Dgooglenet from 1017
    project_variable.experiment_number = 1018
    project_variable.sheet_number = 23
    project_variable.device = 0
    project_variable.end_epoch = 100
    project_variable.batch_size = 20
    project_variable.batch_size_val_test = 20

    project_variable.load_model = [1017, 25, 21, 0]  # exp, model, epoch, run
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

    main_file.run(project_variable)


# project_variable = ProjectVariable(debug_mode=True)
project_variable = ProjectVariable(debug_mode=False)


# e1015_3D_ucf101() # crashed on erdi-home
# e1016_3D_ucf101()     # godel
# e1017_3D_ucf101()   # erdi-home, crashed, continue as 1018
e1018_3D_ucf101()