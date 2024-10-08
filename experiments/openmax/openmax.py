import torch
import numpy as np
import pickle
from utils.configs import config
from openmax.evt_fitting import weibull_tail_fitting
from openmax.openmax_utils import *
from ultralytics import YOLO
import os
from src.utils.clf_utils import *
from src.utils.classification import MosquitoClassifier, XceptionClassifier

class OpenMaxYOLOCLIP:
    def __init__(self, yolo_path, clip_path, train_dl,
                 fold: str,
                 n_class: int,
                 openmax = True,
                 alpha_rank: int = 5,
                 tailsize: int = 25, device: str = "cuda:0",
                 ):
        '''maybe tailsize = n_class be good, or not.'''
    
        self.yolo = YOLO(yolo_path, task="detect")
        self.net = MosquitoClassifier.load_from_checkpoint(
            clip_path, head_version=7, map_location=torch.device(device)
        )
        self.dev = torch.device('cuda:0')
        self.fold = fold
        self.n_class = n_class
        self.alpha_rank = alpha_rank
        self.tailsize = tailsize
        self.labels = range(self.n_class)
        self.best_thresh = 0.0
        self.net.to(self.dev)
        self.net.eval()
        self.openmax = openmax
        # fitting weibull
        if self.openmax:
            print("Fitting weibull models...")
            create_model(self.net, train_dl, self.fold, self.labels, self.tailsize)

    def compute(self, dl):
        openmax = []
        prob_u = []
        y_true = []

        with torch.no_grad():
            for data in dl:
                x = data[0].to(self.dev)
                y = data[1].cpu().numpy()
                # x = prep_CLIP(self.yolo, x) 
                out = self.net(x).cpu().numpy() # clip_model.predict(x)

                for logits, label in zip(out, y):
                    temp_openmax, temp_prob_u = compute_openmax(
                        logits, self.fold, self.alpha_rank, self.labels)
                    openmax.append(temp_openmax)
                    prob_u.append(temp_prob_u)
                    y_true.append(label)
        
        
        openmax = np.asarray(openmax)
        prob_u = np.asarray(prob_u)
        y_true = np.asarray(y_true)
        y_true_bin = get_bin_labels(y_true)
        roc = compute_roc(y_true_bin, prob_u)
        roc_thresh = roc['thresholds']

        best_idx = np.argmax(roc['tpr'] - roc['fpr'])
        best_thresh = roc_thresh[best_idx] 
        self.best_thresh = best_thresh
        return openmax, best_thresh

    def predict(self, dl):
        if self.openmax:
            y_pred = []
            openmax, thresh = self.compute(dl)
            print("thres: ", thresh)
            

            for scores in openmax:
                temp = get_openmax_predict_int(scores, thresh)
                y_pred.append(temp)

            y_pred = np.asarray(y_pred)
            return y_pred
        else:
            print("not implemented yet")

    
    def predict_single(self, x, threshold=None):
        if self.openmax: 
            with torch.no_grad():
                x = x.to(self.dev)
                out = self.net(x).cpu().detach().numpy().reshape(-1)
                
                openmax, _ = compute_openmax(out, self.fold, self.alpha_rank, self.labels)
                if threshold is None:
                    y_pred = get_openmax_predict_int(openmax, self.best_thresh)
                else:
                    y_pred = get_openmax_predict_int(openmax, threshold) 
            return y_pred
        else: 
            with torch.no_grad():
                x = x.to(self.dev)
                out = self.net(x).cpu().detach().numpy().reshape(-1)
                y_pred = np.argmax(out)
            return y_pred

    
    def predict_om_prob(self, x):
        with torch.no_grad():
            x = x.to(self.dev)
            out = self.net(x).cpu().detach().numpy().reshape(-1)            
                       
            openmax, _ = compute_openmax(out, self.fold, self.alpha_rank, self.labels)
        return openmax
# -----------------------------------


class OpenMaxYOLOXception:
    def __init__(self, yolo_path, xception_path, train_dl,
                 fold: str,
                 n_class: int,
                 alpha_rank: int = 4,
                 tailsize: int = 20, device: str = "cuda:0",
                 ):
        '''maybe tailsize = n_class be good, or not.'''
    
        self.yolo = YOLO(yolo_path, task="detect")
        self.net = XceptionClassifier.load_from_checkpoint(
            xception_path, map_location=torch.device(device)
        )
        self.dev = torch.device('cuda:0')
        self.fold = fold
        self.n_class = n_class
        self.alpha_rank = alpha_rank
        self.tailsize = tailsize
        self.labels = range(self.n_class)
        self.best_thresh = 0.0
        self.net.to(self.dev)
        self.net.eval()

        # fitting weibull
        print("Fitting weibull models...")
        create_model(self.net, train_dl, self.fold, self.labels, self.tailsize)

    def compute(self, dl):
        openmax = []
        prob_u = []
        y_true = []

        with torch.no_grad():
            for data in dl:
                x = data[0].to(self.dev)
                y = data[1].cpu().numpy()
                out = self.net(x).cpu().numpy()

                for logits, label in zip(out, y):
                    temp_openmax, temp_prob_u = compute_openmax(
                        logits, self.fold, self.alpha_rank, self.labels)
                    openmax.append(temp_openmax)
                    prob_u.append(temp_prob_u)
                    y_true.append(label)
        
        
        openmax = np.asarray(openmax)
        prob_u = np.asarray(prob_u)
        y_true = np.asarray(y_true)
        print(np.unique(y_true))
        y_true_bin = get_bin_labels(y_true)
        print(y_true_bin)
        roc = compute_roc(y_true_bin, prob_u)
        roc_thresh = roc['thresholds']

        best_idx = np.argmax(roc['tpr'] - roc['fpr'])
        best_thresh = roc_thresh[best_idx] # could be a factor for low acc
        self.best_thresh = best_thresh
        return openmax, best_thresh

    def predict(self, dl):
        y_pred = []
        openmax, thresh = self.compute(dl)
        

        for scores in openmax:
            temp = get_openmax_predict_int(scores, thresh)
            y_pred.append(temp)

        y_pred = np.asarray(y_pred)
        return y_pred
    
    def predict_single(self, x, threshold=None):
        if self.openmax:
            with torch.no_grad():
                x = x.to(self.dev)
                out = self.net(x).cpu().detach().numpy().reshape(-1)
                
                openmax, _ = compute_openmax(out, self.fold, self.alpha_rank, self.labels)
                if threshold is None:
                    y_pred = get_openmax_predict_int(openmax, self.best_thresh)
                else:
                    y_pred = get_openmax_predict_int(openmax, threshold) 
            return y_pred
        else:
            x = prepCls2(x, self.yolo, img_size=(299, 299)) # xception
            with torch.no_grad():
                x = x.to(self.dev)
                out = self.net(x).cpu().detach().numpy().reshape(-1)
                y_pred = np.argmax(out)
            return y_pred

    def predict_om_prob(self, x):
        with torch.no_grad():
            x = x.to(self.dev)
            out = self.net(x).cpu().detach().numpy().reshape(-1)            
                    
            openmax, _ = compute_openmax(out, self.fold, self.alpha_rank, self.labels)
        return openmax
# ---------------------------------------

def create_model(net, dataloader, fold, labels, tailsize):

    # no need to fit anymore if the file is there 
    weibull_path = f'weibull_models/weibull_model_{fold}.pkl'
    if os.path.exists(weibull_path):
        with open(weibull_path, 'rb') as file:
            weibull_model = pickle.load(file)
        
        print(f"Weibull model already fitted, fetching at {weibull_path}")
        return weibull_model
    
    device = torch.device('cuda:0')
    logits_correct = []
    label_correct = []
    print("Fitting Weibulls...")
    with torch.no_grad():
        for data in dataloader:
            x = data[0].to(device)
            y = data[1].cpu().numpy()
            out = net(x).cpu().numpy()
            correct_index = get_correct_classified(y, out)
            out_correct = out[correct_index]
            y_correct = y[correct_index]

            for logits, label in zip(out_correct, y_correct):
                logits_correct.append(logits)
                label_correct.append(label)

    logits_correct = np.asarray(logits_correct)
    label_correct = np.asarray(label_correct)

    av_map = {}
    for label in labels:
        av_map[label] = logits_correct[label_correct == label]

    feature_mean = []
    feature_distance = []
    for label in labels:
        mean = compute_mean_vector(av_map[label])
        distance = compute_distance_dict(mean, av_map[label])
        feature_mean.append(mean)
        feature_distance.append(distance)

    weibull_model = build_weibull(
        mean=feature_mean,
        distance=feature_distance,
        tail=tailsize,
        fold=fold,
        labels = labels
    )
    return weibull_model


def build_weibull(mean, distance, tail, fold, labels):
    model_path = f'weibull_models/weibull_model_{fold}.pkl'
    weibull_model = {}

    for label in labels:
        weibull_model[label] = {}
        weibull = weibull_tail_fitting(
            mean[label], distance[label], tailsize=tail)
        weibull_model[label] = weibull

    with open(model_path, 'wb') as file:
        pickle.dump(weibull_model, file)
    return weibull_model


def query_weibull(label, weibull_model):
    return weibull_model[label]


def recalibrate_scores(weibull_model, activation_vector, alpha_rank, labels):
    ranked_list = np.flip(np.argsort(activation_vector))
    alpha_weights = [
        ((alpha_rank+1) - i) / float(alpha_rank) for i in range(1, alpha_rank+1)
    ]
    ranked_alpha = np.zeros_like(activation_vector)
    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]
    ranked_alpha *= 0.5

    openmax_scores = []
    openmax_scores_u = []

    for label in labels:
        weibull = query_weibull(label, weibull_model)
        av_distance = compute_distance(weibull['mean'], activation_vector)
        wscore = weibull['weibull_model'].w_score(av_distance)
        modified_score = activation_vector[label] * \
            (1 - wscore*ranked_alpha[label])
        openmax_scores += [modified_score]
        openmax_scores_u += [activation_vector[label] - modified_score]

    openmax_scores = np.asarray(openmax_scores)
    openmax_scores_u = np.asarray(openmax_scores_u)

    openmax_prob, prob_u = compute_openmax_probability(
        openmax_scores, openmax_scores_u)
    return openmax_prob, prob_u


def compute_openmax_probability(openmax_scores, openmax_scores_u):
    e_k = np.exp(openmax_scores)
    e_u = np.exp(np.sum(openmax_scores_u))
    openmax_arr = np.concatenate((e_k, e_u), axis=None)
    total_denominator = np.sum(openmax_arr)
    prob_k = e_k / total_denominator
    prob_u = e_u / total_denominator
    res = np.concatenate((prob_k, prob_u), axis=None)
    return res, prob_u


def compute_openmax(activation_vector, fold, alpha_rank, labels):
    model_path = f'weibull_models/weibull_model_{fold}.pkl'
    with open(model_path, 'rb') as file:
        weibull_model = pickle.load(file)
    openmax, prob_u = recalibrate_scores(weibull_model, activation_vector, alpha_rank, labels)
    return openmax, prob_u

