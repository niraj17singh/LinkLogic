import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np


figtype = 'png' 

def get_categorical_palette(plot=False):
    palette = sns.color_palette('colorblind')
    colornames = ['col1', 'Heuristic90', 'col2', 'Heuristic95', 'col3', 'col4', 'col5', 'generic','col6', 'linklogic']
    palette = dict(zip(colornames, palette))
    if plot:
        sns.palplot(palette.values())
    return palette

def get_sequential_palette(plot=False, n=None, name='flare'):
    palette = sns.color_palette(name, n_colors=n)
    
    if plot:
        sns.palplot(palette)
    return palette


def scores_per_path_category(fdata, score_column, experiments=None, title='', 
                             path_column='path category', palette="colorblind", 
                             min_val=0.01, abs_mean=False, query_triple=None, filename=None, 
                             hue='experiment'):
    
    if experiments is not None:
        fdata = fdata[fdata['experiment'].isin(experiments)]
        
    if query_triple is not None:
        fdata = fdata[fdata['query_triple']==query_triple]

    category_means = fdata.groupby(path_column)[score_column].mean()
    
    
    if min_val:
        if abs_mean:
            category_means = category_means[abs(category_means) >= min_val]
        else:
            category_means = category_means[category_means >= min_val]
        ctg_to_keep = list(category_means.index)
        fdata = fdata[fdata[path_column].isin(ctg_to_keep)]
    
    category_order = list(category_means.sort_values(ascending=False).index)

    plt.figure(figsize=(4, 10))

    sns.set(style="white")
    p = sns.barplot(x=score_column, y=path_column, data=fdata, palette=palette, #hue=hue,
                    order=category_order, errwidth=2)#, fliersize=1) hue_order=['with parent--child path', 'without parent--child path']
    _ = p.set(ylabel=None)
    _ = p.legend(loc='lower right')
    p.set_title(title)
    
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        
    plt.show()


def incidents_per_path_category(fdata, palette, experiments=None, score_column='coefficient', update_path_ctg=False,
                                path_column = 'bmk category', min_count=20, topk=None, percent=True, figsize=(5,12),
                                filename=None):
    
    if experiments is not None:
        fdata = fdata[fdata['experiment'].isin(experiments)]

    fdata['positive'] = fdata[score_column] > 0
    fdata_summary = fdata.groupby(['experiment', 'query_triple',path_column], as_index=False)['positive'].any()
    fdata_summary = fdata_summary.groupby(['experiment', path_column], as_index=False)['positive'].sum()

    if percent:
        num_triples = len(set(fdata['query_triple']))
        fdata_summary['percent'] = 100 * (fdata_summary['positive'] / num_triples)
        xcol1 = 'percent'
        xcol2 = '% explanations with path'
    else:
        xcol1 = 'positive'
        xcol2 = 'number explanations with path'

    if min_count:
        fdata_summary = fdata_summary[fdata_summary['positive']>=min_count]

    if update_path_ctg:
        fdata_summary[path_column] = fdata_summary[path_column].map(lambda x: clean_path_category(x))
        
    category_means = fdata_summary.groupby(path_column)[xcol1].mean()
    category_order = list(category_means.sort_values(ascending=False).index)
    
    if topk:
        category_order = category_order[:topk]
    
    sns.set(font_scale=1.5, style='whitegrid')
    plt.figure(figsize=figsize)
    p = sns.barplot(data=fdata_summary, x=xcol1, y=path_column, hue='experiment', order=category_order, palette=palette, hue_order=experiments)
    _ = p.legend(loc='lower right')
    p.set_xlabel(xcol2)
    p.set_ylabel(None)

    
    if percent:
        _ = p.set(xlim=(0, 100))
    
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        
        
def clean_path_category(path):
    replace_punct = [('->', '--'), ('_', '-'), ('--', ', '), ('**', '')]
    replace_letters = [('has-', ''), ('CP', '$\mathregular{p_2}$'), ('C', 'c'), ('P', 'p'), ('S', 's'), ('X', 'x')]
    replace_all = replace_punct + replace_letters
    for pair in replace_all:
        path = path.replace(*pair)
    return '(' + path + ')'


def scatter_with_facets(M, x, y, color, palette='matter', filename=None, size=8, width=1200, height=300):
    fig = px.scatter(M, x=x, y=y, color=color, facet_col='experiment', 
                     hover_data=["query triple", "kge score"],
                     color_continuous_scale=palette, opacity=0.5, width=width, height=height)
    fig.add_shape(dict(type="line", x0=0, y0=0, x1=1.1, y1=1.1, line_width=1),
                       row="all", col="all",
                  line=dict(color="gray", width=1, dash="dot"))
    
    fig.update_traces(marker_size=size, marker_line={'width':0.5, 'color':'lightgrey'})
    
    fig.layout.plot_bgcolor = '#f0f0f0'
    fig.show()
    if filename is not None:
        fig.write_html(filename)