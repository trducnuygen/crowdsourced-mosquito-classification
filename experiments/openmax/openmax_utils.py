import numpy as np
import scipy.spatial.distance as spd
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

N_CLASS = 4 # no mosquito

def compute_mean_vector(feature):
    return np.mean(feature, axis=0)


def compute_distance(mean_feature, feature, distance_type='eucos'):
    if distance_type == 'eucos':
        query_distance = spd.euclidean(
            mean_feature, feature)/200. + spd.cosine(mean_feature, feature)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mean_feature, feature)/200.
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mean_feature, feature)
    else:
        print('Distance type unknown, valid distance type: eucos, euclidean, cosine')
    return query_distance


def compute_distance_dict(mean_feature, feature):
    eu_dist, cos_dist, eucos_dist = [], [], []
    for feat in feature:
        eucos_dist += [spd.euclidean(mean_feature, feat)/200. + spd.cosine(
            mean_feature, feat)]
        eu_dist += [spd.euclidean(mean_feature, feat)/200.]
        cos_dist += [spd.cosine(mean_feature, feat)]
    distances = {'eucos': eucos_dist, 'cosine': cos_dist, 'euclidean': eu_dist}
    return distances


def get_openmax_predict_int(openmax, threshold):
    max_idx = np.argmax(openmax)
    if (max_idx == N_CLASS) or (openmax[N_CLASS] >= threshold):
        return N_CLASS
    else:
        return max_idx


def get_openmax_predict_bin(openmax, threshold):
    max_idx = np.argmax(openmax)
    if (max_idx == N_CLASS) or (openmax[N_CLASS] >= threshold):
        return 1
    else:
        return 0


def get_int_labels(labels):
    labels = np.where(labels > N_CLASS - 1, N_CLASS, labels)
    return labels


def get_bin_labels(labels):
    # unk = 1, k=0
    labels = np.where(labels >= N_CLASS - 1, 1, 0)
    return labels


def get_correct_classified(y_true, y_hat):
    y_true = np.asarray(y_true)
    y_hat = np.argmax(y_hat, axis=1)
    res = y_hat == y_true
    return res


def compute_roc(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)
    roc = {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auroc': auroc,
    }
    return roc


def compute_pr(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    f1 = np.nan_to_num((2*precision*recall) / (precision+recall))
    pr = {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
        'aupr': aupr,
    }
    return pr


def plot_roc(roc):
    plt.plot(roc['fpr'], roc['tpr'])
    plt.grid('on')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUROC: {:.5f}'.format(roc['auroc']))


def plot_pr(pr):
    plt.plot(pr['recall'], pr['precision'])
    plt.grid('on')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUPR: {:.5f}'.format(pr['aupr']))
