from utilities import visualization as VZ


def add_standard_info(project_variable, which, parameters):
    loss, accuracy, confusion_epoch = parameters

    project_variable.writer.add_scalar('loss/%s' % which, loss, project_variable.current_epoch)
    project_variable.writer.add_scalar('accuracy/%s' % which, accuracy, project_variable.current_epoch)
    # fig = VZ.plot_confusion_matrix(confusion_epoch, project_variable.dataset)
    # project_variable.writer.add_figure(tag='confusion/%s' % which, figure=fig, global_step=project_variable.current_epoch)
