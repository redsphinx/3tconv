import numpy as np


class ProjectVariable(object):
    def __init__(self, debug_mode=True):
        # values: lovelace, godel
        self._server = 'lovelace'

        print("\nRUNNING ON '%s' SERVER\n" % self._server)
        if debug_mode:
            print("running in debug mode")

        """
        Default values for all the experimental variables.
        """
        self._writer = None

        self._debug_mode = debug_mode

        # int, which gpu to use {None, 0, 1, etc}
        self._device = None

        # list of int, which model to load, [experiment, model, epoch]
        self._load_model = None
        # only saves the model from the best run
        self._save_only_best_run = True
        # saves all the models
        self._save_all_models = False

        # which model to load. mapping in legend.txt
        self._model_number = None
        # int, experiment data for log
        self._experiment_number = None
        # run in inference mode only on the entire test split
        self._inference_only_mode = False
        # [batch mode: True or False, end_run, expnum, modelnum]
        self._inference_in_batches = [False, None, None, None]
        # for zeiler2014 in xai_tools
        self._return_ind = False
        # to run xai only mode
        self._xai_only_mode = False


        # which google sheet to write to
        self._sheet_number = None

        # bool
        self._pretrain_resnet18_weights = True

        # int, the current epoch
        self._current_epoch = None

        # UNUSED? ================================================================================
        # list of str, which datasets to train, val and test on
        # implemented sets: omg_emotion, affectnet, dhg, jester, tiny_jester, mov_mnist, mnist, kth_actions, ucf101
        self._dataset_train = ['omg_emotion']
        self._dataset_val = ['omg_emotion']
        self._dataset_test = ['omg_emotion']
        # UNUSED? ================================================================================

        # instead of having 3 dataset splits, have just 1 dataset parameter
        # implemented datasets: omg_emotion, mnist, dummy, mov_mnist, kth_actions, dhg, jester  status affectnet??
        self._dataset = 'mnist'
        self._randomize_training_data = False
        self._balance_training_data = False
        self._same_training_data = False
        # if use_dali, these don't have any effect
        # if use_dali, control the number of data using dali_iterator_size
        self._data_points = [100, 100, 100]  # [train, val, test]
        self._use_dali = False
        self._dali_workers = 4
        self._load_from_fast = False

        # default is to use all of the data
        # this controls the number of steps: steps = dali_iterator_size / batch_size
        self._dali_iterator_size = ['all', 'all', 'all']
        self._imnet_mean = None
        self._imnet_std = None
        # for training in parallel
        # self._run_in_parallel = False

        # bool, which procedures to perform
        self._train = None
        self._test = None

        # list of str, which labels to use.
        # omg_emotion: ['categories', 'arousal', 'valence']
        # affect_net: [categories, arousal, valence, face, landmarks]
        self._label_type = ['categories']

        # int, label size for the output type
        # omg_emotion categories: 7
        # affectnet categories: 11
        # mnist: 10
        # dhg: 14
        # jester: 27
        self._label_size = 10

        # float, learning rate
        self._learning_rate = 0.001
        # use cyclical learning rate
        self._use_clr = False
        # str, loss function
        self._loss_function = 'cross_entropy'
        # list, weights for balanced loss, necessary for resnet18
        self._loss_weights = None
        # list of str, optimizer
        # supported: adam, sgd
        self._optimizer = 'sgd'
        # momentum
        self._momentum = 0.9

        # number of out_channels in the convolution layers of CNNs
        self._num_out_channels = [6, 16]

        # number of color channels in the input
        self._num_in_channels = 3

        # int, seed for shuffling
        self._seed = 6

        # depending on debug mode
        if debug_mode:
            self._batch_size = 30
            self._batch_size_val_test = 30
            self._start_epoch = -1
            self._end_epoch = 5
            self._train_steps = 10
            self._val_steps = 1
            self._test_steps = 1
            self._save_data = False
            self._save_model = False
            self._save_graphs = False
        else:
            self._batch_size = 30
            self._batch_size_val_test = 30
            self._start_epoch = -1
            self._end_epoch = 20
            self._train_steps = 50
            self._val_steps = 10
            self._test_steps = 10
            self._save_data = True
            self._save_model = True
            self._save_graphs = True

        # if set to 1, saves every epoch
        # if set to None, it saves according to the other params
        self.save_model_every_x_epochs = None  

        self._repeat_experiments = 1
        self._at_which_run = 0
        # how to initialize experiment files and saves: 'new': new experiment, 'crashed': experiment crashed before
        # finishing, 'extra': experiment finished, run an additional batch of the same experiment
        self._experiment_state = 'new'
        # which dataset to evaluate on
        self._eval_on = 'test'

        # ----------------------------------------------------------------------------------------------------------
        # settings only for 3dconvttn stuff
        # ----------------------------------------------------------------------------------------------------------
        # how to initialize theta: 'normal', 'eye', 'eye-like' or None. if None, theta is created from affine params
        self._theta_init = 'eye'
        # how to initialize SRXY: 'normal', 'eye'=[1,0,0,0], 'eye-like'=[1+e,e,e,e]
        self._srxy_init = 'normal'
        # how to transform weights in kernel: 'naive'=weights are a transform of first_weight, 'seq'=sequential
        self._weight_transform = 'naive'
        # which kind of smoothness constraint for the srxy values: None, 'sigmoid', 'sigmoid_small'
        self._srxy_smoothness = None
        # k_0 initialization: 'normal', 'ones', 'ones_var'=mean=1,std=0.5, 'uniform'
        self._k0_init = 'normal'
        # share transformation parameters across all filters in a layer.
        # her_e we set how many sets of transformations are learned.
        # note that 1 <= transformation_groups <= num_out_channels
        self._transformation_groups = self.num_out_channels # across filters
        # filters share k0
        self._k0_groups = self.num_out_channels
        # shape of convolution filter
        self._k_shape = (5, 5, 5)
        # transformation groups within t of a single filter
        self._transformations_per_filter = self.k_shape[0] - 1
        # time dimension of the 3D max pooling
        self._max_pool_temporal = 2
        # height=width dimension of the convolutional kernels
        self._conv_k_hw = 3
        # train only the temporal parameters in the last layer
        self._only_theta_final_layer = False
        # ----------------------------------------------------------------------------------------------------------
        # setting for video datasets
        # ----------------------------------------------------------------------------------------------------------
        self._load_num_frames = 30
        # time dimension of the kernel in conv1
        self._conv1_k_t = 3
        # where to add batchnorm after each non-linear activation layer
        self._do_batchnorm = [False, False, False, False, False]
        # ----------------------------------------------------------------------------------------------------------
        # settings for adaptive learning rate
        # ----------------------------------------------------------------------------------------------------------
        self._use_adaptive_lr = False
        self._adapt_eval_on = 'val'
        self._reduction_factor = 2
        self._decrease_after_num_epochs = 10
        # ----------------------------------------------------------------------------------------------------------
        # separate learning rate for theta
        # ----------------------------------------------------------------------------------------------------------
        self._theta_learning_rate = None
        # set to True for k0 to also have theta_lr
        self._k0_theta_learning_rate = False
        # set to True for bias to also have theta_lr
        self._bias_theta_learning_rate = False

        # ----------------------------------------------------------------------------------------------------------
        # stuff for the CLR: https://arxiv.org/pdf/1506.01186.pdf
        # ----------------------------------------------------------------------------------------------------------
        # self._use_clr = False
        # self._num_cycles = 2
        # self._min_learning_rate = 1e-5
        # self._max_learning_rate = 1e-2
        # self._mode = 'triangle'

        # ----------------------------------------------------------------------------------------------------------
        # XAI stuff for visualization
        # ----------------------------------------------------------------------------------------------------------
        self._do_xai = False
        # options: 'erhan2009', 'zeiler2014', 'gradient_method'
        self._which_methods = ['erhan2009']
        # options: 'conv1', 'conv2'
        self._which_layers = ['conv1']
        # options: [np.arange(6), np.arange(16)]
        # each index maps to the respective layer
        self._which_channels = [np.array([0, 1])]

        # ----------------------------------------------------------------------------------------------------------
        # NAS stuff, optimized training
        # ----------------------------------------------------------------------------------------------------------
        # turn on or off semi-automated architecture search
        self._nas = False
        # stop when confusion matrix has collapsed
        self._stop_at_collapse = False
        # stop when validation accuracy is going down + validation loss is going up
        self._early_stopping = False
        # for GA in NAS
        self._genome = None
        # to identify which individual it is in the population
        self._individual_number = None

        # ----------------------------------------------------------------------------------------------------------
        # NiN idea
        # ----------------------------------------------------------------------------------------------------------
        self._nin = False
        # valid options: 'joint', 'nin_only', 'alternating', False
        # if False, nin does not get trained
        self._train_nin_mode = 'joint'

        # ----------------------------------------------------------------------------------------------------------
        # generating dots dataset
        # ----------------------------------------------------------------------------------------------------------
        self._dots_mode = False

    @property
    def writer(self):
        return self._writer

    @writer.setter
    def writer(self, value):
        self._writer = value

    @property
    def debug_mode(self):
        return self._debug_mode

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value

    @property
    def load_model(self):
        return self._load_model

    @load_model.setter
    def load_model(self, value):
        self._load_model = value

    @property
    def save_only_best_run(self):
        return self._save_only_best_run

    @save_only_best_run.setter
    def save_only_best_run(self, value):
        self._save_only_best_run = value

    @property
    def save_all_models(self):
        return self._save_all_models

    @save_all_models.setter
    def save_all_models(self, value):
        self._save_all_models = value

    @property
    def model_number(self):
        return self._model_number

    @model_number.setter
    def model_number(self, value):
        self._model_number = value

    @property
    def experiment_number(self):
        return self._experiment_number

    @experiment_number.setter
    def experiment_number(self, value):
        self._experiment_number = value

    @property
    def inference_only_mode(self):
        return self._inference_only_mode

    @inference_only_mode.setter
    def inference_only_mode(self, value):
        self._inference_only_mode = value

    @property
    def inference_in_batches(self):
        return self._inference_in_batches

    @inference_in_batches.setter
    def inference_in_batches(self, value):
        self._inference_in_batches = value

    @property
    def return_ind(self):
        return self._return_ind

    @return_ind.setter
    def return_ind(self, value):
        self._return_ind = value

    @property
    def xai_only_mode(self):
        return self._xai_only_mode

    @xai_only_mode.setter
    def xai_only_mode(self, value):
        self._xai_only_mode = value


    @property
    def sheet_number(self):
        return self._sheet_number

    @sheet_number.setter
    def sheet_number(self, value):
        self._sheet_number = value

    @property
    def pretrain_resnet18_weights(self):
        return self._pretrain_resnet18_weights

    @pretrain_resnet18_weights.setter
    def pretrain_resnet18_weights(self, value):
        self._pretrain_resnet18_weights = value

    @property
    def current_epoch(self):
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, value):
        self._current_epoch = value

    @property
    def dataset_train(self):
        return self._dataset_train

    @dataset_train.setter
    def dataset_train(self, value):
        self._dataset_train = value

    @property
    def dataset_val(self):
        return self._dataset_val

    @dataset_val.setter
    def dataset_val(self, value):
        self._dataset_val = value

    @property
    def dataset_test(self):
        return self._dataset_test

    @dataset_test.setter
    def dataset_test(self, value):
        self._dataset_test = value

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value

    @property
    def randomize_training_data(self):
        return self._randomize_training_data

    @randomize_training_data.setter
    def randomize_training_data(self, value):
        self._randomize_training_data = value

    @property
    def balance_training_data(self):
        return self._balance_training_data

    @balance_training_data.setter
    def balance_training_data(self, value):
        self._balance_training_data = value

    @property
    def same_training_data(self):
        return self._same_training_data

    @same_training_data.setter
    def same_training_data(self, value):
        self._same_training_data = value

    @property
    def data_points(self):
        return self._data_points

    @data_points.setter
    def data_points(self, value):
        self._data_points = value

    @property
    def use_dali(self):
        return self._use_dali

    @use_dali.setter
    def use_dali(self, value):
        self._use_dali = value

    @property
    def dali_workers(self):
        return self._dali_workers

    @dali_workers.setter
    def dali_workers(self, value):
        self._dali_workers = value

    @property
    def load_from_fast(self):
        return self._load_from_fast

    @load_from_fast.setter
    def load_from_fast(self, value):
        self._load_from_fast = value

    @property
    def server(self):
        return self._server

    @server.setter
    def server(self, value):
        self._server = value

    @property
    def dali_iterator_size(self):
        return self._dali_iterator_size

    @dali_iterator_size.setter
    def dali_iterator_size(self, value):
        self._dali_iterator_size = value

    @property
    def imnet_mean(self):
        return self._imnet_mean

    @imnet_mean.setter
    def imnet_mean(self, value):
        self._imnet_mean = value

    @property
    def imnet_std(self):
        return self._imnet_std

    @imnet_std.setter
    def imnet_std(self, value):
        self._imnet_std = value

    # @property
    # def run_in_parallel(self):
    #     return self._run_in_parallel
    #
    # @run_in_parallel.setter
    # def run_in_parallel(self, value):
    #     self._run_in_parallel = value

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, value):
        self._train = value

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, value):
        self._val = value

    @property
    def test(self):
        return self._test

    @test.setter
    def test(self, value):
        self._test = value

    @property
    def label_type(self):
        return self._label_type

    @label_type.setter
    def label_type(self, value):
        self._label_type = value

    @property
    def label_size(self):
        return self._label_size

    @label_size.setter
    def label_size(self, value):
        self._label_size = value

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value

    @property
    def use_clr(self):
        return self._use_clr

    @use_clr.setter
    def use_clr(self, value):
        self._use_clr = value

    @property
    def loss_function(self):
        return self._loss_function

    @loss_function.setter
    def loss_function(self, value):
        self._loss_function = value

    @property
    def loss_weights(self):
        return self._loss_weights

    @loss_weights.setter
    def loss_weights(self, value):
        self._loss_weights = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def momentum(self):
        return self._momentum

    @momentum.setter
    def momentum(self, value):
        self._momentum = value

    @property
    def num_out_channels(self):
        return self._num_out_channels

    @num_out_channels.setter
    def num_out_channels(self, value):
        self._num_out_channels = value

    @property
    def num_in_channels(self):
        return self._num_in_channels

    @num_in_channels.setter
    def num_in_channels(self, value):
        self._num_in_channels = value

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @property
    def batch_size_val_test(self):
        return self._batch_size_val_test

    @batch_size_val_test.setter
    def batch_size_val_test(self, value):
        self._batch_size_val_test = value

    @property
    def start_epoch(self):
        return self._start_epoch

    @start_epoch.setter
    def start_epoch(self, value):
        self._start_epoch = value

    @property
    def end_epoch(self):
        return self._end_epoch

    @end_epoch.setter
    def end_epoch(self, value):
        self._end_epoch = value

    @property
    def train_steps(self):
        return self._train_steps

    @train_steps.setter
    def train_steps(self, value):
        self._train_steps = value

    @property
    def val_steps(self):
        return self._val_steps

    @val_steps.setter
    def val_steps(self, value):
        self._val_steps = value

    @property
    def test_steps(self):
        return self._test_steps

    @test_steps.setter
    def test_steps(self, value):
        self._test_steps = value

    @property
    def save_data(self):
        return self._save_data

    @save_data.setter
    def save_data(self, value):
        self._save_data = value

    @property
    def save_model(self):
        return self._save_model

    @save_model.setter
    def save_model(self, value):
        self._save_model = value

    @property
    def save_graphs(self):
        return self._save_graphs

    @save_graphs.setter
    def save_graphs(self, value):
        self._save_graphs = value


    @property
    def save_model_every_x_epochs(self):
        return self._save_model_every_x_epochs
    
    @save_model_every_x_epochs.setter
    def save_model_every_x_epochs(self, value):
        self._save_model_every_x_epochs = value

    @property
    def repeat_experiments(self):
        return self._repeat_experiments

    @repeat_experiments.setter
    def repeat_experiments(self, value):
        self._repeat_experiments = value

    @property
    def at_which_run(self):
        return self._at_which_run

    @at_which_run.setter
    def at_which_run(self, value):
        self._at_which_run = value

    @property
    def experiment_state(self):
        return self._experiment_state

    @experiment_state.setter
    def experiment_state(self, value):
        self._experiment_state = value

    @property
    def eval_on(self):
        return self._eval_on

    @eval_on.setter
    def eval_on(self, value):
        self._eval_on = value

    @property
    def theta_init(self):
        return self._theta_init

    @theta_init.setter
    def theta_init(self, value):
        self._theta_init = value

    @property
    def srxy_init(self):
        return self._srxy_init

    @srxy_init.setter
    def srxy_init(self, value):
        self._srxy_init = value

    @property
    def weight_transform(self):
        return self._weight_transform

    @weight_transform.setter
    def weight_transform(self, value):
        self._weight_transform = value

    @property
    def srxy_smoothness(self):
        return self._srxy_smoothness

    @srxy_smoothness.setter
    def srxy_smoothness(self, value):
        self._srxy_smoothness = value

    @property
    def k0_init(self):
        return self._k0_init

    @k0_init.setter
    def k0_init(self, value):
        self._k0_init = value

    @property
    def transformation_groups(self):
        return self._transformation_groups

    @transformation_groups.setter
    def transformation_groups(self, value):
        self._transformation_groups = value

    @property
    def k0_groups(self):
        return self._k0_groups

    @k0_groups.setter
    def k0_groups(self, value):
        self._k0_groups = value

    @property
    def k_shape(self):
        return self._k_shape

    @k_shape.setter
    def k_shape(self, value):
        self._k_shape = value

    @property
    def transformations_per_filter(self):
        return self._transformations_per_filter

    @transformations_per_filter.setter
    def transformations_per_filter(self, value):
        self._transformations_per_filter = value

    @property
    def max_pool_temporal(self):
        return self._max_pool_temporal

    @max_pool_temporal.setter
    def max_pool_temporal(self, value):
        self._max_pool_temporal = value

    @property
    def conv_k_hw(self):
        return self._conv_k_hw

    @conv_k_hw.setter
    def conv_k_hw(self, value):
        self._conv_k_hw = value

    @property
    def load_num_frames(self):
        return self._load_num_frames

    @load_num_frames.setter
    def load_num_frames(self, value):
        self._load_num_frames = value

    @property
    def conv1_k_t(self):
        return self._conv1_k_t

    @conv1_k_t.setter
    def conv1_k_t(self, value):
        self._conv1_k_t = value

    @property
    def only_theta_final_layer(self):
        return self._only_theta_final_layer

    @only_theta_final_layer.setter
    def only_theta_final_layer(self, value):
        self._only_theta_final_layer = value

    @property
    def do_batchnorm(self):
        return self._do_batchnorm

    @do_batchnorm.setter
    def do_batchnorm(self, value):
        self._do_batchnorm = value

    @property
    def use_adaptive_lr(self):
        return self._use_adaptive_lr

    @use_adaptive_lr.setter
    def use_adaptive_lr(self, value):
        self._use_adaptive_lr = value

    @property
    def adapt_eval_on(self):
        return self._adapt_eval_on

    @adapt_eval_on.setter
    def adapt_eval_on(self, value):
        self._adapt_eval_on = value

    @property
    def reduction_factor(self):
        return self._reduction_factor

    @reduction_factor.setter
    def reduction_factor(self, value):
        self._reduction_factor = value

    @property
    def decrease_after_num_epochs(self):
        return self._decrease_after_num_epochs

    @decrease_after_num_epochs.setter
    def decrease_after_num_epochs(self, value):
        self._decrease_after_num_epochs = value

    @property
    def theta_learning_rate(self):
        return self._theta_learning_rate

    @theta_learning_rate.setter
    def theta_learning_rate(self, value):
        self._theta_learning_rate = value

    @property
    def k0_theta_learning_rate(self):
        return self._k0_theta_learning_rate

    @k0_theta_learning_rate.setter
    def k0_theta_learning_rate(self, value):
        self._k0_theta_learning_rate = value

    @property
    def bias_theta_learning_rate(self):
        return self._bias_theta_learning_rate

    @bias_theta_learning_rate.setter
    def bias_theta_learning_rate(self, value):
        self._bias_theta_learning_rate = value

    @property
    def do_xai(self):
        return self._do_xai

    @do_xai.setter
    def do_xai(self, value):
        self._do_xai = value

    @property
    def which_methods(self):
        return self._which_methods

    @which_methods.setter
    def which_methods(self, value):
        self._which_methods = value

    @property
    def which_layers(self):
        return self._which_layers

    @which_layers.setter
    def which_layers(self, value):
        self._which_layers = value

    @property
    def which_channels(self):
        return self._which_channels

    @which_channels.setter
    def which_channels(self, value):
        self._which_channels = value

    @property
    def nas(self):
        return self._nas

    @nas.setter
    def nas(self, value):
        self._nas = value

    @property
    def stop_at_collapse(self):
        return self._stop_at_collapse

    @stop_at_collapse.setter
    def stop_at_collapse(self, value):
        self._stop_at_collapse = value

    @property
    def early_stopping(self):
        return self._early_stopping

    @early_stopping.setter
    def early_stopping(self, value):
        self._early_stopping = value

    @property
    def genome(self):
        return self._genome

    @genome.setter
    def genome(self, value):
        self._genome = value

    @property
    def individual_number(self):
        return self._individual_number

    @individual_number.setter
    def individual_number(self, value):
        self._individual_number = value

    @property
    def nin(self):
        return self._nin
    
    @nin.setter
    def nin(self, value):
        self._nin = value

    @property
    def train_nin_mode(self):
        return self._train_nin_mode
    
    @train_nin_mode.setter
    def train_nin_mode(self, value):
        self._train_nin_mode = value

    @property
    def dots_mode(self):
        return self._dots_mode

    @dots_mode.setter
    def dots_mode(self, value):
        self._dots_mode = value