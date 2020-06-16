import numpy as np
import torch
from tqdm import tqdm

from utilities import utils as U
from utilities import tensorboard as TM
from utilities import saving
from data import data_loading as DL


def run(project_variable, all_data, my_model, my_optimizer, device):

    loss_epoch, accuracy_epoch, confusion_epoch, nice_div, steps, full_labels, full_data = \
        U.initialize(project_variable, all_data)

    if project_variable.use_dali:
        if project_variable.dataset == 'jester':
            the_iterator = DL.get_jester_iter('train', project_variable)
        elif project_variable.dataset == 'ucf101':
            the_iterator = DL.get_ucf101_iter('train', project_variable)
        else:
            the_iterator = None

        for i, data_and_labels in tqdm(enumerate(the_iterator)):

            data = data_and_labels[0]['data']
            labels = data_and_labels[0]['labels']

            # transpose data
            data = data.permute(0, 4, 1, 2, 3)
            # convert to floattensor
            data = data.type(torch.float32)

            # data shape: b, c, d, h, w
            data = data / 255
            data[:, 0, :, :, :] = (data[:, 0, :, :, :] - 0.485) / 0.229
            data[:, 1, :, :, :] = (data[:, 1, :, :, :] - 0.456) / 0.224
            data[:, 2, :, :, :] = (data[:, 2, :, :, :] - 0.406) / 0.225

            # data = (data/255 - project_variable.imnet_mean) / project_variable.imnet_stds
            labels = labels.type(torch.long)
            labels = labels.flatten()
            if project_variable.dataset == 'jester':
                labels = labels - 1

            my_optimizer.zero_grad()

            if project_variable.model_number in [20]:
                predictions = my_model(data, device)
            elif project_variable.model_number in [23]:
                aux1, aux2, predictions = my_model(data, device)
                assert aux1 is not None and aux2 is not None
            elif project_variable.model_number in [25]:
                aux1, aux2, predictions = my_model(data)
                assert aux1 is not None and aux2 is not None

            else:
                predictions = my_model(data)


            if project_variable.model_number in [23, 25]:
                loss = U.googlenet_loss(project_variable, aux1, aux2, predictions, labels)
            else:
                loss = U.calculate_loss(project_variable, predictions, labels)
            # THCudaCheck FAIL file=/pytorch/aten/src/THC/THCGeneral.cpp line=383 error=11 : invalid argument
            loss.backward()

            my_optimizer.step()


            accuracy = U.calculate_accuracy(predictions, labels)
            confusion_epoch = U.confusion_matrix(confusion_epoch, predictions, labels)

            loss_epoch.append(float(loss))
            accuracy_epoch.append(float(accuracy))


    # save data
    loss = float(np.mean(loss_epoch))


    if project_variable.use_dali:
        accuracy = sum(accuracy_epoch) / (steps * project_variable.batch_size)
    else:
        accuracy = sum(accuracy_epoch) / (steps * project_variable.batch_size + nice_div)

    confusion_flatten = U.flatten_confusion(confusion_epoch)


    if project_variable.save_data:
        saving.update_logs(project_variable, 'train', [loss, accuracy, confusion_flatten])

    print('epoch %d train, %s: %f, accuracy: %f ' % (project_variable.current_epoch,
                                                     project_variable.loss_function,
                                                     loss, accuracy))

    # save model
    if project_variable.save_model:
        if project_variable.stop_at_collapse or project_variable.early_stopping:
            saving.save_model(project_variable, my_model)

        else:
            if project_variable.current_epoch == project_variable.end_epoch - 1:
                saving.save_model(project_variable, my_model)

    # add things to writer
    TM.add_standard_info(project_variable, 'train', (loss, accuracy, confusion_epoch))


    if project_variable.nas or project_variable.stop_at_collapse:
        return accuracy, U.has_collapsed(confusion_epoch)
    else:
        return accuracy