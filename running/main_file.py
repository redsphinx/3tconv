import os
import time
import numpy as np
import torch
from tensorboardX import SummaryWriter
import shutil
import math


from running import training, validation, testing
from utilities import setup
from utilities import utils as U
from utilities import sheets as S
from data import data_loading as D
from config import paths as PP




def run(project_variable):
    # torch.autograd.set_detect_anomaly(True)

    START_LR = project_variable.learning_rate
    if project_variable.theta_learning_rate is not None:
        START_LR_THETA = project_variable.theta_learning_rate

    # write initial settings to spreadsheet
    if not project_variable.debug_mode:
        if project_variable.experiment_state == 'new':
            ROW = S.write_settings(project_variable)
        elif project_variable.experiment_state == 'crashed':
            project_variable.at_which_run = U.experiment_exists(project_variable.experiment_number,
                                                                project_variable.model_number)
            ROW = S.get_specific_row(project_variable.experiment_number, project_variable.sheet_number)
        elif project_variable.experiment_state == 'extra':
            project_variable.at_which_run = 1 + U.experiment_exists(project_variable.experiment_number,
                                                                    project_variable.model_number)
            project_variable.repeat_experiments += project_variable.at_which_run
            ROW = S.get_specific_row(project_variable.experiment_number, project_variable.sheet_number)

    # remove duplicate log files
    log_file = 'experiment_%d_model_%d_run_%d.txt' % (project_variable.experiment_number,
                                                      project_variable.model_number,
                                                      project_variable.at_which_run)
    _which = ['train', 'val', 'test']
    for w in _which:
        log_path = os.path.join(PP.saving_data, w, log_file)
        if os.path.exists(log_path):
            if os.path.isdir(log_path):
                shutil.rmtree(log_path)
            else:
                os.remove(log_path)

    start = project_variable.at_which_run

    if project_variable.inference_only_mode:
        if project_variable.eval_on == 'val':
            project_variable.val = True
        else:
            project_variable.val = False
    else:
        project_variable.val = True

    if project_variable.eval_on == 'test':
        project_variable.test = True
    else:
        project_variable.test = False

    if project_variable.inference_only_mode:
        project_variable.train = False
    else:
        if not project_variable.randomize_training_data:
            project_variable.train = True
        else:
            project_variable.train = False

    # setup model, optimizer & device
    my_model = setup.get_model(project_variable)
    print(U.count_parameters(my_model))
    device = setup.get_device(project_variable)

    if project_variable.device is not None:
        my_model.cuda(device)

    if project_variable.inference_only_mode:
        pass
    else:
        my_optimizer = setup.get_optimizer(project_variable, my_model)

    print('Loaded model number %d with %d trainable parameters' % (
        project_variable.model_number, U.count_parameters(my_model)))

    # create the dali iterators
    if not project_variable.use_dali:

        data = D.load_data(project_variable, seed=None)

        if project_variable.train:
            data_train = D.get_data('train', data)
            labels_train = D.get_labels('train', data)
        if project_variable.val:
            data_val = D.get_data('val', data)
            labels_val = D.get_labels('val', data)
        if project_variable.test:
            data_test = D.get_data('test', data)
            labels_test = D.get_labels('test', data)

        # to ensure the same data will be chosen between various models
        # useful when experimenting with low number of datapoints
        if project_variable.same_training_data:
            np.random.seed(project_variable.data_points)
            # each run has a unique seed based on the initial datapoints configuration
            training_seed_runs = np.random.randint(10000, size=project_variable.repeat_experiments)

    # keep track of how many runs have collapsed and at which epoch it stops training
    runs_collapsed = np.zeros(shape=project_variable.repeat_experiments, dtype=int)
    which_epoch_stopped = np.ones(shape=project_variable.repeat_experiments, dtype=int) * -1

    # ====================================================================================================
    # start with runs
    # ====================================================================================================
    val_accuracy = None

    for num_runs in range(start, project_variable.repeat_experiments):
        if not project_variable.use_dali:
            if project_variable.same_training_data:
                seed = training_seed_runs[num_runs]
            else:
                seed = None

        if not project_variable.use_dali:
            # load the training data (which is now randomized)
            if not project_variable.inference_only_mode:
                if project_variable.randomize_training_data:
                    project_variable.test = False
                    project_variable.val = False
                    project_variable.train = True
                    data = D.load_data(project_variable, seed)
                    if project_variable.train:
                        data_train = D.get_data('train', data)
                        labels_train = D.get_labels('train', data)

        print('-------------------------------------------------------\n\n'
              'RUN: %d / %d\n\n'
              '-------------------------------------------------------'
              % (num_runs, project_variable.repeat_experiments))

        # create writer for tensorboardX
        if not project_variable.debug_mode:
            path = os.path.join(PP.writer_path, 'experiment_%d_model_%d' % (project_variable.experiment_number,
                                                                            project_variable.model_number))
            subfolder = os.path.join(path, 'run_%d' % project_variable.at_which_run)

            path = subfolder

        else:
            path = os.path.join(PP.writer_path, 'debugging')

        if not project_variable.nas:
            if not os.path.exists(path):
                os.makedirs(path)
            else:
                # clear directory before writing new events
                shutil.rmtree(path)
                time.sleep(2)
                os.mkdir(path)

        project_variable.writer = SummaryWriter(path)
        print('tensorboardX writer path: %s' % path)



        if not project_variable.debug_mode:
            if num_runs == 0:
                S.write_parameters(U.count_parameters(my_model), ROW, project_variable.sheet_number)

        # add project settings to writer
        text = 'experiment number:      %d;' \
               'model number:           %d;' \
               'trainable parameters:   %d;' \
               % (project_variable.experiment_number,
                  project_variable.model_number,
                  U.count_parameters(my_model)
                  )
        project_variable.writer.add_text('project settings', text)

        # ====================================================================================================
        # start with epochs
        # ====================================================================================================

        # setup performance tracking for adapting learning rate
        # index 0   which epoch to check
        # index 1   the performance metric
        if project_variable.use_adaptive_lr:
            reduction_epochs = project_variable.end_epoch // project_variable.decrease_after_num_epochs
            track_performance = np.zeros((reduction_epochs))
            check_epochs = []
            for r in range(reduction_epochs):
                check_epochs.append(r * project_variable.decrease_after_num_epochs)
        else:
            check_epochs = None
            track_performance = None

        # keeping track of collapsed training confusion matrix
        # if there is collapse for 3 times in a row, we stop the experiment
        collapse_limit = 3
        collapse_tracker = 0

        # keeping track of validation accuracy for early stopping
        check_every_num_epoch = 5
        checking_at_epochs = [9 + (i * check_every_num_epoch) for i in
                              range(project_variable.end_epoch // check_every_num_epoch)]
        checking_at_epochs = [0] + checking_at_epochs
        val_acc_tracker = 0
        val_loss_tracker = math.inf

        # variable depending on settings 'stop_at_collapse=True' and/or 'early_stopping=True'
        stop_experiment = False

        for e in range(project_variable.start_epoch + 1, project_variable.end_epoch):
            if stop_experiment:
                break

            else:
                if project_variable.inference_only_mode:
                    pass
                else:
                    print('--------------------------------------------------------------------------\n'
                          'STARTING LEARNING RATE: %s\n'
                          '--------------------------------------------------------------------------'
                          % str(project_variable.learning_rate))

                    if project_variable.theta_learning_rate is not None:
                        print('--------------------------------------------------------------------------\n'
                              'STARTING THETA LEARNING RATE: %s\n'
                              '--------------------------------------------------------------------------'
                              % str(project_variable.theta_learning_rate))

                project_variable.current_epoch = e

                # get data
                # splits = ['train', 'val', 'test']
                # final_data = [[img0, img1,...],
                #               [img0, img1,...],
                #               [img0, img1,...]]
                # final_labels = [[arousal, valence, categories],
                #                 [arousal, valence, categories],
                #                 [arousal, valence, categories]]

                # ------------------------------------------------------------------------------------------------
                # ------------------------------------------------------------------------------------------------
                # HERE: TRAINING
                # ------------------------------------------------------------------------------------------------
                # ------------------------------------------------------------------------------------------------
                if project_variable.inference_only_mode or project_variable.xai_only_mode:
                    project_variable.train = False
                else:
                    project_variable.train = True
                project_variable.val = False
                project_variable.test = False



                if project_variable.train:
                    condition_1 = project_variable.dataset == 'jester' and project_variable.use_dali and not project_variable.nas
                    condition_2 = not project_variable.debug_mode and project_variable.dataset == 'jester'

                    w = None
                    if condition_1 or condition_2:
                        w = np.array([0.0379007, 0.03862456, 0.0370375, 0.03737979, 0.03620443,
                                      0.03648918, 0.03675273, 0.03750421, 0.03627937, 0.03738865,
                                      0.03676129, 0.03696806, 0.03817587, 0.0391227, 0.04904935,
                                      0.04642222, 0.0371072, 0.03684716, 0.0366673, 0.03648918,
                                      0.03607197, 0.03593228, 0.0370462, 0.03637139, 0.03608847,
                                      0.036873, 0.01644524])

                    if w is not None:
                        w = w.astype(dtype=np.float32)
                        w = torch.from_numpy(w).cuda(device)
                        project_variable.loss_weights = w

                    # labels is list because can be more than one type of labels
                    if project_variable.use_dali:
                        data = None
                        my_model.train()
                    else:
                        data = data_train, labels_train
                        my_model.train()


                    if project_variable.nas or project_variable.stop_at_collapse:
                        train_accuracy, (has_collapsed, collapsed_matrix) = training.run(project_variable, data,
                                                                                         my_model, my_optimizer, device)
                    else:
                        train_accuracy = training.run(project_variable, data, my_model, my_optimizer, device)


                # ------------------------------------------------------------------------------------------------
                # ------------------------------------------------------------------------------------------------
                # HERE: VALIDATION
                # ------------------------------------------------------------------------------------------------
                # ------------------------------------------------------------------------------------------------
                if project_variable.inference_only_mode:
                    if project_variable.eval_on == 'val':
                        project_variable.val = True
                    else:
                        project_variable.val = False
                elif project_variable.xai_only_mode:
                    project_variable.val = False
                else:
                    project_variable.val = True

                project_variable.train = False
                project_variable.test = False

                if project_variable.val:
                    condition_1 = project_variable.dataset == 'jester' and project_variable.use_dali and not project_variable.nas
                    condition_2 = not project_variable.debug_mode and project_variable.dataset == 'jester'

                    w = None
                    if condition_1 or condition_2:
                        w = np.array([0.03913151, 0.03897372, 0.03606524, 0.03929058, 0.03464331, 0.0377558,
                                      0.03945095, 0.03913151, 0.03593117, 0.03593117, 0.03835509, 0.04044135,
                                      0.03731847, 0.0370325, 0.04931369, 0.04646867, 0.0354047, 0.03760889,
                                      0.03620031, 0.0377558, 0.03675089, 0.03593117, 0.03620031, 0.03451958,
                                      0.03464331, 0.03647352, 0.01327676])
                    if w is not None:
                        w = w.astype(dtype=np.float32)
                        w = torch.from_numpy(w).cuda(device)
                        project_variable.loss_weights = w

                    if project_variable.use_dali:
                        data = None
                    else:
                        data = data_val, labels_val

                    if project_variable.early_stopping:
                        val_accuracy, val_loss = validation.run(project_variable, data, my_model, device)
                    else:
                        # print('!! val accuracy updated !!')
                        val_accuracy = validation.run(project_variable, data, my_model, device)
                # ------------------------------------------------------------------------------------------------
                # ------------------------------------------------------------------------------------------------
                # HERE: TESTING
                # ------------------------------------------------------------------------------------------------
                # ------------------------------------------------------------------------------------------------
                # only run at the last epoch
                if e == project_variable.end_epoch - 1 or project_variable.inference_only_mode:
                    # project_variable.train = False
                    # project_variable.val = False
                    if project_variable.eval_on == 'test':
                        project_variable.test = True
                    else:
                        project_variable.test = False
                else:
                    project_variable.test = False


                    if project_variable.test:
                        condition_1 = project_variable.dataset == 'jester' and project_variable.use_dali and not project_variable.nas
                        condition_2 = not project_variable.debug_mode and project_variable.dataset == 'jester'

                        w = None
                        if condition_1 or condition_2:
                            w = np.array([0.03897281, 0.04044657, 0.03819954, 0.03674154, 0.03716712, 0.0356529,
                                          0.03513242, 0.03578544, 0.03674154, 0.03804855, 0.03450281, 0.03437958,
                                          0.03674154, 0.0414926, 0.05093271, 0.05039939, 0.03804855, 0.03578544,
                                          0.03775013, 0.03513242, 0.03487784, 0.03605349, 0.03688231, 0.03760267,
                                          0.03760267, 0.03591897, 0.01300849])
                        if w is not None:
                            w = w.astype(dtype=np.float32)
                            w = torch.from_numpy(w).cuda(device)
                            project_variable.loss_weights = w

                        if project_variable.use_dali:
                            data = None
                        else:
                            data = data_test, labels_test

                        testing.run(project_variable, data, my_model, device)
                # ------------------------------------------------------------------------------------------------
                # ------------------------------------------------------------------------------------------------

                # at the end of an epoch
                if project_variable.use_adaptive_lr:
                    if e in check_epochs:
                        idx = check_epochs.index(e)
                        # update epoch with performance information
                        if project_variable.adapt_eval_on == 'train':
                            track_performance[idx] = train_accuracy
                        elif project_variable.adapt_eval_on == 'val':
                            track_performance[idx] = val_accuracy
                        if idx > 0:
                            if track_performance[idx] > track_performance[idx - 1]:
                                pass
                            else:
                                print('--------------------------------------------------------------------------\n'
                                      'LEARNING RATE REDUCED: from %s to %s\n'
                                      '--------------------------------------------------------------------------'
                                      % (str(project_variable.learning_rate),
                                         str(project_variable.learning_rate / project_variable.reduction_factor)))

                                project_variable.learning_rate /= project_variable.reduction_factor

                                if project_variable.theta_learning_rate is not None:
                                    print('--------------------------------------------------------------------------\n'
                                          'THETA LEARNING RATE REDUCED: from %s to %s\n'
                                          '--------------------------------------------------------------------------'
                                          % (str(project_variable.theta_learning_rate),
                                             str(
                                                 project_variable.theta_learning_rate / project_variable.reduction_factor)))

                                    project_variable.theta_learning_rate /= project_variable.reduction_factor

                # decide if the experiment should stop
                if project_variable.stop_at_collapse and project_variable.early_stopping:

                    if has_collapsed:
                        collapse_tracker = collapse_tracker + 1
                        if collapse_tracker >= collapse_limit:
                            stop_experiment = True
                            runs_collapsed[num_runs] = 1
                            which_epoch_stopped[num_runs] = e

                    elif project_variable.current_epoch in checking_at_epochs:
                        if val_accuracy < val_acc_tracker and val_loss > val_loss_tracker:
                            stop_experiment = True
                            which_epoch_stopped[num_runs] = e
                        val_acc_tracker = val_accuracy
                        val_loss_tracker = val_loss

                elif project_variable.stop_at_collapse or project_variable.nas:
                    if has_collapsed:
                        collapse_tracker = collapse_tracker + 1
                        if collapse_tracker >= collapse_limit:
                            stop_experiment = True
                            runs_collapsed[num_runs] = 1
                            which_epoch_stopped[num_runs] = e

                elif project_variable.early_stopping:
                    if project_variable.current_epoch in checking_at_epochs:
                        if val_accuracy < val_acc_tracker and val_loss > val_loss_tracker:
                            stop_experiment = True
                            which_epoch_stopped[num_runs] = e
                        val_acc_tracker = val_accuracy
                        val_loss_tracker = val_loss

        # at the end of a run
        project_variable.at_which_run += 1
        project_variable.writer.close()
        project_variable.learning_rate = START_LR  # reset the learning rate
        if project_variable.theta_learning_rate is not None:
            project_variable.theta_learning_rate = START_LR_THETA  # reset theta learning rate

    if not project_variable.debug_mode:
        acc, std, best_run = U.experiment_runs_statistics(project_variable.experiment_number,
                                                          project_variable.model_number, mode=project_variable.eval_on)

        S.write_results(acc, std, best_run, ROW, project_variable.sheet_number)

        if project_variable.stop_at_collapse or project_variable.early_stopping:
            best_run_stopped = which_epoch_stopped[best_run]
            num_runs_collapsed = sum(runs_collapsed)
            S.extra_write_results(int(best_run_stopped), int(num_runs_collapsed), ROW, project_variable.sheet_number)

        if project_variable.save_only_best_run:
            U.delete_runs(project_variable, best_run)

    print('\n\n\n END OF EXPERIMENT %d \n\n\n' % (project_variable.experiment_number))

    if project_variable.nas:
        return project_variable.individual_number, has_collapsed, val_accuracy, train_accuracy

    if project_variable.dots_mode:
        return val_accuracy

# from config.base_config import ProjectVariable
# project_variable = ProjectVariable(debug_mode=True)
# project_variable.experiment_number = 1014
# project_variable.model_number = 23
# project_variable.eval_on = 'val'
# print(U.experiment_runs_statistics(project_variable.experiment_number,
#                              project_variable.model_number, mode=project_variable.eval_on))