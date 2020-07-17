import numpy as np
import torch
from tqdm import tqdm

from utilities import utils as U
from utilities import tensorboard as TM
from utilities import saving
from data import data_loading as DL


def run(project_variable, all_data, my_model, device):

    loss_epoch, accuracy_epoch, confusion_epoch, nice_div, steps, full_labels, full_data = \
        U.initialize(project_variable, all_data)

    if project_variable.use_dali:
        if project_variable.dataset == 'jester':
            the_iterator = DL.get_jester_iter('test', project_variable)
        elif project_variable.dataset == 'tiny_jester':
            the_iterator = DL.get_tiny_jester_iter('test', project_variable)
        elif project_variable.dataset == 'ucf101':
            the_iterator = DL.get_ucf101_iter('test', project_variable)
        else:
            the_iterator = None
        steps = 0

        for i, data_and_labels in enumerate(the_iterator):

            data = data_and_labels[0]['data']
            labels = data_and_labels[0]['labels']

            # transpose data
            data = data.permute(0, 4, 1, 2, 3)
            # convert to floattensor
            data = data.type(torch.float32)
            labels = labels.type(torch.long)
            labels = labels.flatten()
            if 'jester' in project_variable.dataset:
                labels = labels - 1

            my_model.eval()
            with torch.no_grad():
                if project_variable.model_number in [23]:
                    aux1, aux2, predictions = my_model(data, device, None, False)
                    assert aux1 is None and aux2 is None
                elif project_variable.model_number in [20, 51, 53]:
                    predictions = my_model(data, device)

                elif project_variable.model_number in [25]:
                    aux1, aux2, predictions = my_model(data, None, False)
                    assert aux1 is None and aux2 is None

                elif project_variable.model_number in [50, 52, 54]:
                    predictions = my_model(data, device, og_datapoint=data)

                else:
                    predictions = my_model(data)
                # print(predictions)
                loss = U.calculate_loss(project_variable, predictions, labels)
                loss = loss.detach()

            my_model.train()


    # save data
    # print('loss epoch: ', loss_epoch)
    loss = float(np.mean(loss_epoch))
    if project_variable.use_dali:
        accuracy = sum(accuracy_epoch) / (steps * project_variable.batch_size)
    else:
        accuracy = sum(accuracy_epoch) / (steps * project_variable.batch_size + nice_div)

    confusion_flatten = U.flatten_confusion(confusion_epoch)

    if project_variable.save_data:
        saving.update_logs(project_variable, 'test', [loss, accuracy, confusion_flatten])

    print('epoch %d test, %s: %f, accuracy: %f ' % (project_variable.current_epoch,
                                                    project_variable.loss_function,
                                                    loss, accuracy))

    TM.add_standard_info(project_variable, 'test', (loss, accuracy, confusion_epoch))


    if project_variable.inference_in_batches[0]:
        return accuracy