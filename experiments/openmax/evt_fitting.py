import numpy as np
import libmr


def weibull_tail_fitting(mean, distance, tailsize, distance_type='eucos'):
    weibull_model = {}
    distance_scores = np.asarray(distance[distance_type])
    mean_scores = np.array(mean)

    tail = sorted(distance_scores)[-tailsize:]
    mr = libmr.MR()
    mr.fit_high(tail, len(tail))

    weibull_model['mean'] = mean_scores
    weibull_model['weibull_model'] = mr
    return weibull_model
