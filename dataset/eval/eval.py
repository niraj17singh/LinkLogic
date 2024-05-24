import pandas as pd
import math
import utils as ut
from sklearn.metrics import ndcg_score
import numpy as np

def get_results_for_query_triple(results, triple):
    out = None
    for i, res in enumerate(results):
        if res['query_triple'] == triple:
            out = res
            break
    return out

def extract_feature_df_from_results(res, names, bmk_df=None, score_column='coefficient', kthresh=5, **kwargs):
    rows = []
    
    if bmk_df is not None:
        bmk_paths = set(bmk_df.path)
    else:
        bmk_paths = []
    
    for f in res:
        path = ut.stringify_path(f['path'])
        path_score = f['kge_score']['path_score']
        coef = f['coef']
        label = int(path in bmk_paths)
        str_label = '**' if label else ''
        rows.append({'path': path,
                     'path category': str_label + categorize_path(path, names),
                     'coefficient': coef,
                     'coefficient product': coef*path_score,
                     'baseline path score': path_score,
                     'label': label})
    df = pd.DataFrame(rows)

    df['normalized ' + score_column] = compute_norm_score(df, score_column=score_column)
    df[f'top{kthresh} path score'] = threshold_score(df, column='baseline path score', k=kthresh)

    df['random'] = 0
    
    if bmk_df is not None:
        df = df.merge(bmk_df, on='path', how='left')
        df['bmk confidence'] = df['bmk confidence'].fillna(0)
        df['bmk category'] = df['bmk category'].fillna('non-benchmark')        

    for key, value in kwargs.items():
        df[key] = value

    return df


def categorize_path(path, names):
    
    path = normalize_path_for_category(path, names)

    slist = []
    slist.append(letter_name(names, path[0]))
    slist.append(rel_string(path[1]))
    slist.append(letter_name(names, path[2]))
    
    if len(path) == 5:
        slist.append(rel_string(path[3]))
        slist.append(letter_name(names, path[4]))
    elif len(path) != 3:
        raise ValueError('unexpected length of path')
        
    return ''.join(slist)

def letter_name(names, name, filler='X'):
    letters = {
        'child': 'C',
        'parent': 'P',
        'coparent': 'CP',
        'siblings': 'S',
        'sibling': 'S'
    }
    assert filler not in letters.keys()
    
    rev_names = reverse_entity_names(names)
    title = rev_names.get(name, filler)
    letter = letters.get(title, filler)
    return letter

def reverse_entity_names(names):
    rev_names = {}
    for key, values in names.items():
        if type(values) == str:
            values = [values]
        if values is not None:
            for value in values:
                rev_names[value] = key
    return rev_names


def normalize_path_for_category(path, names):
    path = path.split('--')
    
    if len(path) == 3:
        path_out = path
    else:
        assert len(path) == 5
        if path[0] == names['child']:
            assert path[4] == names['parent']
            path_out = path
        else:
            assert path[4] == names['child'] and path[0] == names['parent']
            path_out = (path[4], flip_rel(path[3]), path[2], flip_rel(path[1]), path[0])
    return path_out
        
        
def flip_rel(rel):
    if rel == 'parents':
        rel_flip = 'children'
    elif rel == 'children':
        rel_flip = 'parents'
    else:
        rel_flip = rel
    return rel_flip


def rel_string(rel):
    if rel == 'parents':
        s = '--has_parent->'
    elif rel == 'children':
        s = '--has_child->'
    else:
        s = f'--{rel}--'
    return s


def compute_norm_score(df, score_column='coefficient'):
    max_val = df[score_column].max()
    denominator = max_val if max_val > 0 else 1
    return df[score_column] / denominator

def threshold_score(df, column, k):
    df_sorted = df.sort_values(by=column, ascending=False).reset_index()
    thresh = df_sorted.loc[k-1,column]
    out = df[column]
    out[out < thresh] = 0
    assert (out>0).value_counts()[True] == k
    return out 

def extract_bmk_paths_as_df(bmk, ctg_column='category'):
    rows = []
    for p in bmk['explanatory_paths']:
        row = {'path': ut.stringify_path(p['path']),
               'bmk confidence': p['confidence'],
               'bmk category': p[ctg_column]}
        rows.append(row)
    return pd.DataFrame(rows)

def mean_diffs_by_confidence(df):
    if has_multi_confidence_bmks(df):
        levels = set(df.bmk_confidence).difference({0})
        assert len(levels) == 2
        low_confidence = str(min(levels))
        high_confidence = str(max(levels))
        all_means = mean_by_confidence(df, column='coefficient')
        diff = all_means[high_confidence] - all_means[low_confidence]
    else:
        diff = math.nan
    return diff

def mean_by_confidence(df, column='coefficient'):
    levels = set(df.bmk_confidence)
    means = dict()
    for level in levels:
        means[str(level)] = df.loc[df.bmk_confidence == level, column].mean()
    return means

def has_multi_confidence_bmks(df):
    return len(set(df.bmk_confidence).difference({0})) > 1

def get_bmk_results(df):
    return df[df.label == 1]

def get_explanation_stats(df, rank_by='coefficient', **kwargs):
    df['rank'] = df[rank_by].rank(method='max', ascending=False)
    out = df.groupby('bmk category', as_index=False)
    out = out[[rank_by, 'rank']].mean()
    for key, value in kwargs.items():
        out[key] = value
    return out

def ndcg_score_range(y_true, y_score, k_range):
    return [ndcg_score(y_true=y_true, y_score=y_score, k=k) for k in k_range]

def precision_at_k(y_true, y_score, k):
    df = pd.DataFrame({'true': y_true, 'estimate':y_score})
    df.sort_values(by='estimate', ascending=False, inplace=True)
    df= df.iloc[:k]
    return df['true'].sum() / k

def precision_at_k_range(y_true, y_score, k_range):
    return [precision_at_k(y_true, y_score, k=k) for k in k_range]

def filter_k_range(k_range, series):
    max_k = np.count_nonzero(series)
    if max_k < k_range[-1]:
        k_range = k_range[:max_k]
    return k_range