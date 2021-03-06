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
        the_iterator = DL.get_iterator('val', project_variable)

        steps = 0

        for i, data_and_labels in tqdm(enumerate(the_iterator)):

            data = data_and_labels[0]['data']  # torch.Size([30, 30, 150, 224, 3])
            labels = data_and_labels[0]['labels']


            if len(data.shape) == 5:
                # transpose data
                data = data.permute(0, 4, 1, 2, 3)  # torch.Size([30, 3, 30, 150, 224])
                # convert to floattensor
                data = data.type(torch.float32)

                if project_variable.model_number not in [51, 52]:

                    data = data / 255
                    data[:, 0, :, :, :] = (data[:, 0, :, :, :] - 0.485) / 0.229
                    data[:, 1, :, :, :] = (data[:, 1, :, :, :] - 0.456) / 0.224
                    data[:, 2, :, :, :] = (data[:, 2, :, :, :] - 0.406) / 0.225

            elif len(data.shape) == 4:
                data = data.permute(0, 3, 1, 2)
                # convert to floattensor
                data = data.type(torch.float32)


            labels = labels.type(torch.long)
            labels = labels.flatten()
            if project_variable.dataset in ['jester', 'kinetics400_metaclass']:
                labels = labels - 1

            if not labels.is_cuda:
                labels = labels.cuda(device)

            my_model.eval()
            with torch.no_grad():
                if project_variable.model_number in [23]:
                    aux1, aux2, predictions = my_model(data, device, None, False)
                    assert aux1 is None and aux2 is None
                elif project_variable.model_number in [20, 51, 53, 56, 57]:
                    predictions = my_model(data, device)

                elif project_variable.model_number in [25]:
                    aux1, aux2, predictions = my_model(data, None, False)
                    assert aux1 is None and aux2 is None

                elif project_variable.model_number in [50, 52, 54]:
                    assert project_variable.nin
                    # predictions = my_model(data, device, og_datapoint=data)
                    predictions = my_model(data, device)

                else:
                    predictions = my_model(data)
                # print(predictions)
                loss = U.calculate_loss(project_variable, predictions, labels)
                loss = loss.detach()

            my_model.train()

            accuracy = U.calculate_accuracy(predictions, labels)
            confusion_epoch = U.confusion_matrix(confusion_epoch, predictions, labels)

            loss_epoch.append(float(loss))
            accuracy_epoch.append(float(accuracy))

            steps = steps + 1


    # save data
    # print('loss epoch: ', loss_epoch)
    loss = float(np.mean(loss_epoch))
    if project_variable.use_dali:
        accuracy = sum(accuracy_epoch) / (steps * project_variable.batch_size)
    else:
        accuracy = sum(accuracy_epoch) / (steps * project_variable.batch_size + nice_div)

    confusion_flatten = U.flatten_confusion(confusion_epoch)

    if project_variable.save_data:
        saving.update_logs(project_variable, 'val', [loss, accuracy, confusion_flatten])

    print('epoch %d val, %s: %f, accuracy: %f ' % (project_variable.current_epoch,
                                                   project_variable.loss_function,
                                                   loss, accuracy))

    TM.add_standard_info(project_variable, 'val', (loss, accuracy, confusion_epoch))


    if project_variable.early_stopping:
        return accuracy, loss
    else:
        return accuracy