from config.base_config import ProjectVariable
from running import main_file
from utilities import utils

# kinetics400
# train:  213808
# val:    17243
# test:   ???

def set_init_1():
    project_variable.end_epoch = 100
    project_variable.dataset = 'kinetics400'
    project_variable.sheet_number = 24
    project_variable.num_in_channels = 3
    project_variable.data_points = [2 * 400,  1 * 400, 0 * 400]
    project_variable.label_size = 400
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


def e001_3T_kinetics():
    set_init_1()
    project_variable.model_number = 23 # googlenet
    project_variable.experiment_number = 1

    project_variable.device = 2
    project_variable.end_epoch = 200
    project_variable.batch_size = 18   # 5 about 3000
    project_variable.batch_size_val_test = 30 # 30 about 3000

    # project_variable.inference_only_mode = True
    project_variable.inference_only_mode = False

    # project_variable.save_model_every_x_epochs = 1

    project_variable.load_model = True # loading pre-trained on ImageNet
    project_variable.load_from_fast = True

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.stop_at_collapse = True
    # project_variable.stop_at_collapse = False
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.00005
    project_variable.use_adaptive_lr = True

    main_file.run(project_variable)


# project_variable = ProjectVariable(debug_mode=True)
project_variable = ProjectVariable(debug_mode=False)


e001_3T_kinetics()

