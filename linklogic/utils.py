import numpy as np

def parse_query(query):
    return query[0], query[1], query[2]

def sort_scores(scores, axis=0, prob=True):
    if prob==True:
        return -np.sort(-scores, axis=axis)
    else:
        return np.sort(scores, axis=axis)
def sort_entities(scores, axis=0, prob=True):
    if prob==True:
        return np.argsort(-scores, axis=axis)
    else:
        return np.argsort(scores, axis=axis)

def logsum1(score):
    return -np.log(1.0-score)

def logsum2(score1, score2):
    return (-np.log(1.0-score1) - np.log(1.0-score2))/2.0