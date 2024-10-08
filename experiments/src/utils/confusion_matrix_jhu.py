import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

from sklearn.metrics import confusion_matrix


def plot_pretty_blue_confusion_matrix(classes, y_true=[], y_pred=[], cm=[],
                                      normalize=False,
                                      title=None,
                                      cmap=plt.cm.Blues,
                                      savepath=None,
                                      figsize=(24,24)):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'
    labels=list(range(0,len(classes),1))
    # Compute confusion matrix
    if (len(y_true)==0 and len(y_pred)==0) and len(cm)==0:
        sys.exit('need at least cm or the predictions and truths')
    elif len(cm)==0:
        cm = confusion_matrix(y_true, y_pred,labels=labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        acc = 0
        for i in range(len(classes)):
            acc += cm[i][i]
        acc = acc/len(classes)
        print('Avg Acc',acc)
    else:
        print("Confusion matrix")

    save_df = pd.DataFrame(data=cm, columns=classes)
    save_df.index = classes
    save_df.to_csv(savepath+'.csv', index=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.gcf().subplots_adjust(bottom=0.25)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if math.isnan(val):
                val = 0.0
            ax.text(j, i, format(val, fmt),
                    ha="center", va="center", size=12,
                    color="white" if cm[i, j] > thresh else "black")
    if savepath is not None:
        fig.savefig(savepath + '.jpg',dpi=150)
    return ax
