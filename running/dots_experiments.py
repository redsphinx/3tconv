from config.base_config import ProjectVariable
from running import main_file
from utilities import utils

# dots
# train:  10000
# val:    2000
# test:   5000

def set_init_1_frames():
    project_variable.end_epoch = 100
    project_variable.dataset = 'dots_frames'
    project_variable.sheet_number = 25
    project_variable.num_in_channels = 1
    project_variable.label_size = 3
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


def e001_dots():
    set_init_1_frames()
    project_variable.model_number = 55 # lenet5 2D
    project_variable.experiment_number = 1

    project_variable.device = 0
    project_variable.end_epoch = 10
    project_variable.batch_size = 32
    project_variable.batch_size_val_test = 32

    # project_variable.inference_only_mode = True
    project_variable.inference_only_mode = False

    project_variable.load_model = False # loading model from scratch
    # project_variable.load_from_fast = True  # UNUSED?

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.stop_at_collapse = True
    # project_variable.stop_at_collapse = False
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    # project_variable.learning_rate = 0.05
    project_variable.learning_rate = 0.001
    project_variable.use_adaptive_lr = True

    main_file.run(project_variable)


def e002_dots():
    set_init_1_frames()
    project_variable.dataset = 'dots_frames'
    # project_variable.dataset = 'ucf101'
    # project_variable.model_number = 56 # lenet5 3t
    project_variable.model_number = 55 # lenet5 2D
    project_variable.experiment_number = 2

    project_variable.device = 0
    project_variable.end_epoch = 10
    project_variable.batch_size = 32
    project_variable.batch_size_val_test = 32

    # project_variable.inference_only_mode = True
    project_variable.inference_only_mode = False

    project_variable.load_model = False # loading model from scratch
    # project_variable.load_from_fast = True  # UNUSED?

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    # project_variable.stop_at_collapse = True
    project_variable.stop_at_collapse = False
    # project_variable.early_stopping = True
    project_variable.early_stopping = False

    project_variable.optimizer = 'adam'
    # project_variable.learning_rate = 0.05
    project_variable.learning_rate = 0.0005
    project_variable.use_adaptive_lr = True

    main_file.run(project_variable)


def e003_dots():
    set_init_1_frames()
    project_variable.dataset = 'dots_avi'
    # project_variable.dataset = 'ucf101'
    project_variable.model_number = 56 # lenet5 3t
    # project_variable.model_number = 55 # lenet5 2D
    project_variable.experiment_number = 3
    project_variable.load_num_frames = 15

    project_variable.device = 0
    project_variable.end_epoch = 10
    project_variable.batch_size = 32
    project_variable.batch_size_val_test = 32

    # project_variable.inference_only_mode = True
    project_variable.inference_only_mode = False

    project_variable.load_model = False # loading model from scratch
    # project_variable.load_from_fast = True  # UNUSED?

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    # project_variable.stop_at_collapse = True
    project_variable.stop_at_collapse = False
    # project_variable.early_stopping = True
    project_variable.early_stopping = False

    project_variable.optimizer = 'adam'
    # project_variable.learning_rate = 0.05
    project_variable.learning_rate = 0.00001
    project_variable.use_adaptive_lr = True

    main_file.run(project_variable)


def e004_dots():
    set_init_1_frames()
    project_variable.dataset = 'dots_avi'
    # project_variable.dataset = 'ucf101'
    project_variable.model_number = 56 # lenet5 3t
    # project_variable.model_number = 55 # lenet5 2D
    project_variable.experiment_number = 4
    project_variable.load_num_frames = 15

    project_variable.device = 0
    project_variable.end_epoch = 100
    project_variable.batch_size = 32
    project_variable.batch_size_val_test = 32

    # project_variable.inference_only_mode = True
    project_variable.inference_only_mode = False

    project_variable.load_model = False # loading model from scratch
    # project_variable.load_from_fast = True  # UNUSED?

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    # project_variable.stop_at_collapse = True
    project_variable.stop_at_collapse = False
    # project_variable.early_stopping = True
    project_variable.early_stopping = False

    project_variable.optimizer = 'adam'
    # project_variable.learning_rate = 0.05
    project_variable.learning_rate = 0.00005
    project_variable.use_adaptive_lr = True

    main_file.run(project_variable)


def e005_dots():
    set_init_1_frames()
    project_variable.dataset = 'dots_avi'
    # project_variable.dataset = 'ucf101'
    project_variable.model_number = 56 # lenet5 3t
    # project_variable.model_number = 55 # lenet5 2D
    project_variable.experiment_number = 5
    project_variable.load_num_frames = 15

    project_variable.device = 0
    project_variable.end_epoch = 100
    project_variable.batch_size = 32
    project_variable.batch_size_val_test = 32

    # project_variable.inference_only_mode = True
    project_variable.inference_only_mode = False

    project_variable.load_model = False # loading model from scratch
    # project_variable.load_from_fast = True  # UNUSED?

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    # project_variable.stop_at_collapse = True
    project_variable.stop_at_collapse = False
    # project_variable.early_stopping = True
    project_variable.early_stopping = False

    project_variable.optimizer = 'adam'
    # project_variable.learning_rate = 0.05
    project_variable.learning_rate = 0.0001
    project_variable.use_adaptive_lr = True

    main_file.run(project_variable)


def e006_dots():
    set_init_1_frames()
    project_variable.dataset = 'dots_avi'
    # project_variable.dataset = 'ucf101'
    project_variable.model_number = 56 # lenet5 3t
    # project_variable.model_number = 55 # lenet5 2D
    project_variable.experiment_number = 6
    project_variable.load_num_frames = 15

    project_variable.device = 0
    project_variable.end_epoch = 100
    project_variable.batch_size = 32
    project_variable.batch_size_val_test = 32

    # project_variable.inference_only_mode = True
    project_variable.inference_only_mode = False

    project_variable.load_model = False # loading model from scratch
    # project_variable.load_from_fast = True  # UNUSED?

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.save_all_models = True
    # project_variable.stop_at_collapse = True
    project_variable.stop_at_collapse = False
    # project_variable.early_stopping = True
    project_variable.early_stopping = False

    project_variable.optimizer = 'adam'
    # project_variable.learning_rate = 0.05
    project_variable.learning_rate = 0.001
    project_variable.use_adaptive_lr = True

    main_file.run(project_variable)


def e007_dots():
    set_init_1_frames()
    project_variable.dataset = 'dots_avi'
    # project_variable.dataset = 'ucf101'
    project_variable.model_number = 56 # lenet5 3t
    # project_variable.model_number = 55 # lenet5 2D
    project_variable.experiment_number = 7
    project_variable.load_num_frames = 15

    project_variable.device = 0
    project_variable.end_epoch = 100
    project_variable.batch_size = 32
    project_variable.batch_size_val_test = 32

    # project_variable.inference_only_mode = True
    project_variable.inference_only_mode = False

    project_variable.load_model = False # loading model from scratch
    # project_variable.load_from_fast = True  # UNUSED?

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.save_all_models = True
    # project_variable.stop_at_collapse = True
    project_variable.stop_at_collapse = False
    # project_variable.early_stopping = True
    project_variable.early_stopping = False

    project_variable.optimizer = 'adam'
    # project_variable.learning_rate = 0.05
    project_variable.learning_rate = 0.0003
    project_variable.use_adaptive_lr = True

    main_file.run(project_variable)


def e008_dots():
    set_init_1_frames()
    project_variable.dataset = 'dots_avi'
    # project_variable.dataset = 'ucf101'
    project_variable.model_number = 57 # simplenet3t
    project_variable.experiment_number = 8
    project_variable.load_num_frames = 15

    project_variable.device = 0
    # project_variable.end_epoch = 100
    project_variable.end_epoch = 1
    project_variable.batch_size = 32
    project_variable.batch_size_val_test = 32

    # project_variable.inference_only_mode = True
    project_variable.inference_only_mode = False

    project_variable.load_model = False # loading model from scratch
    # project_variable.load_from_fast = True  # UNUSED?

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.save_all_models = True
    # project_variable.stop_at_collapse = True
    project_variable.stop_at_collapse = False
    # project_variable.early_stopping = True
    project_variable.early_stopping = False

    project_variable.optimizer = 'adam'
    # project_variable.learning_rate = 0.05
    project_variable.learning_rate = 0.0001
    project_variable.use_adaptive_lr = True

    main_file.run(project_variable)


project_variable = ProjectVariable(debug_mode=True)
# project_variable = ProjectVariable(debug_mode=False)

# e001_dots()
# e002_dots()
# e003_dots()
# e004_dots()
# e005_dots()
# e006_dots()
# e007_dots()
e008_dots()