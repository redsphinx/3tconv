import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from textwrap import wrap
import re
import itertools
import numpy as np


def plot_confusion_matrix(confusion_matrix, dataset):

    if dataset == 'jester':
        # labels = ["Swiping Left", "Swiping Right", "Swiping Down", "Swiping Up", "Pushing Hand Away",
        #           "Pulling Hand In", "Sliding Two Fingers Left", "Sliding Two Fingers Right",
        #           "Sliding Two Fingers Down", "Sliding Two Fingers Up", "Pushing Two Fingers Away",
        #           "Pulling Two Fingers In", "Rolling Hand Forward", "Rolling Hand Backward", "Turning Hand Clockwise",
        #           "Turning Hand Counterclockwise", "Zooming In With Full Hand", "Zooming Out With Full Hand",
        #           "Zooming In With Two Fingers", "Zooming Out With Two Fingers", "Thumb Up", "Thumb Down",
        #           "Shaking Hand", "Stop Sign", "Drumming Fingers", "No gesture", "Doing other things"]
        labels = [str(i) for i in range(27)]
    else:
        labels = []

    cm = confusion_matrix

    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    fig = plt.Figure(figsize=(6, 6), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=10, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=12)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=10, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=16, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)
    # summary = tfplot.figure.to_summary(fig, tag=tensor_name)
    return fig