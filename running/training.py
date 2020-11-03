import os
import numpy as np
import torch
from tqdm import tqdm
import cProfile, pstats
from io import StringIO

from utilities import utils as U
from utilities import tensorboard as TM
from utilities import saving
from utilities import visualization as VIS
from data import data_loading as DL
from config import paths as PP



def run(project_variable, all_data, my_model, my_optimizer, device):

    loss_epoch, accuracy_epoch, confusion_epoch, nice_div, steps, full_labels, full_data = \
        U.initialize(project_variable, all_data)

    if project_variable.use_dali:

        the_iterator = DL.get_iterator('train', project_variable)

        steps = 0

        for i, data_and_labels in tqdm(enumerate(the_iterator)):

            data = data_and_labels[0]['data']  # torch.Size([30, 30, 150, 224, 3])
            labels = data_and_labels[0]['labels']

            # if file_root in dataloader, labels are mapped on alphabetic sort

            # HERE----
            # for labels to classes mapping
            # num_im_save = 5
            # for im in range(num_im_save):
            #     label_copy = labels[im].cpu()
            #     data_copy = np.array(data[im].cpu(), dtype=np.uint8)
            #
            #     if data.shape[-1] == 1:
            #         save_path = os.path.join(PP.dots_samples, '%d_label_is_%d.jpg' % (im, label_copy))
            #         VIS.save_array_as_image(data_copy, save_path, 'L')
            #     else:
            #         save_path = os.path.join(PP.dots_samples, '%d_label_is_%d.avi' % (im, label_copy))
            #         VIS.save_array_as_avi(data_copy, save_path)
            # HERE----

            if len(data.shape) == 5:
                # transpose data
                data = data.permute(0, 4, 1, 2, 3)
                # convert to floattensor
                data = data.type(torch.float32)  # torch.Size([1, 3, 30, 150, 224]) # torch.Size([30, 3, 30, 150, 224])

                if project_variable.model_number not in [51, 52]:
                    # if project_variable.nin:
                    #     resized_data = U.resize_data(data.clone())

                    # data shape: b, c, d, h, w
                    data = data / 255
                    data[:, 0, :, :, :] = (data[:, 0, :, :, :] - 0.485) / 0.229
                    data[:, 1, :, :, :] = (data[:, 1, :, :, :] - 0.456) / 0.224
                    data[:, 2, :, :, :] = (data[:, 2, :, :, :] - 0.406) / 0.225

                # if project_variable.nin:
                #     resized_data = resized_data / 255
                #     resized_data[:, 0, :, :, :] = (resized_data[:, 0, :, :, :] - 0.485) / 0.229
                #     resized_data[:, 1, :, :, :] = (resized_data[:, 1, :, :, :] - 0.456) / 0.224
                #     resized_data[:, 2, :, :, :] = (resized_data[:, 2, :, :, :] - 0.406) / 0.225

                # data = (data/255 - project_variable.imnet_mean) / project_variable.imnet_stds

            elif len(data.shape) == 4:
                data = data.permute(0, 3, 1, 2)
                # convert to floattensor
                data = data.type(torch.float32)
            


            labels = labels.type(torch.long)
            labels = labels.flatten()
            if 'jester' in project_variable.dataset:
                labels = labels - 1

            if not labels.is_cuda:
                labels = labels.cuda(device)

            my_optimizer.zero_grad()

            if project_variable.model_number in [20, 51, 53]:
                predictions = my_model(data, device)
            elif project_variable.model_number in [23]:
                aux1, aux2, predictions = my_model(data, device)
                assert aux1 is not None and aux2 is not None
            elif project_variable.model_number in [25]:
                aux1, aux2, predictions = my_model(data)
                assert aux1 is not None and aux2 is not None
            elif project_variable.model_number in [50, 52, 54]:
                assert project_variable.nin
                # predictions = my_model(data, device, resized_datapoint=resized_data)
                predictions = my_model(data, device)

            else:
                predictions = my_model(data)


            if project_variable.model_number in [23, 25]:
                loss = U.googlenet_loss(project_variable, aux1, aux2, predictions, labels)
            else:
                loss = U.calculate_loss(project_variable, predictions, labels)
            # THCudaCheck FAIL file=/pytorch/aten/src/THC/THCGeneral.cpp line=383 error=11 : invalid argument
            # loss.backward(retain_graph=True)
            loss.backward()

            my_optimizer.step()


            accuracy = U.calculate_accuracy(predictions, labels)
            confusion_epoch = U.confusion_matrix(confusion_epoch, predictions, labels)

            loss_epoch.append(float(loss))
            accuracy_epoch.append(float(accuracy))

            steps = steps + 1

            # here
            # pr.disable()
            # s = StringIO()
            # sortby = 'cumulative'
            # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            # ps.print_stats()
            # print(s.getvalue())


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