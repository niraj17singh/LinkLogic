import pickle5 as pickle
import json
from sklearn.metrics import ndcg_score
import pandas as pd
import plotly.express as px

import eval
import utils as ut


def main():

    # settings
    plot_individual_results = False

    # load benchmark
    with open('../../data/commonsense_benchmark/v3/commonsense_benchmark_for_analysis.json', 'r') as f:
        benchmark = json.load(f)

    # load explanations
    results_dir = '../../results/linklogic/num_samples/'
    results = dict()
    n_samples = ['n_1000', 'n_2500', 'n_5000', 'n_7500']
    experiments = ['_fb12', '']

    for experiment in experiments:
        results[experiment] = dict()

        for n in n_samples:
            with open(f'{results_dir}/parents_analysis_ComplEx_all_logsum_True_alpha_0.2_{n}{experiment}.pickle',
                      'rb') as f:
                results[experiment][n] = pickle.load(f)

    # compute metrics
    metrics = []
    colnames = ['bmk_category', 'coefficient', 'rank', 'experiment', 'n_samples', 'query_triple']
    all_stats = pd.DataFrame(columns=colnames)
    feature_df_list = []

    for i, bmk in enumerate(benchmark):

        if bmk['category'] == 'location':
            continue

        triple = bmk['query_triple']
        s_triple = ut.stringify_path(triple)

        bmk_df = eval.extract_bmk_paths_as_df(bmk)

        for experiment in experiments:

            for n in n_samples:

                res = eval.get_results_for_query_triple(results[experiment][n], triple)

                if res:
                    feature_df = eval.extract_feature_df_from_results(res['linklogic_features'], bmk_df,
                                                                      experiment=experiment, n_samples=n, query_triple=s_triple)
                    num_true_candidates = feature_df['label'].sum()

                    if num_true_candidates > 0:

                        feature_df_list.append(feature_df)

                        #stats = eval.get_explanation_stats(feature_df, experiment=experiment, n_samples=n,
                        #                                   query_triple=s_triple)

                        #all_stats = pd.concat([all_stats, feature_df], axis=0)

                        y_true = [feature_df['label']]
                        fidelity_train = res['linklogic_metrics']['train_acc']
                        fidelity_test = res['linklogic_metrics']['test_acc']
                        kge_score = float(res['query_triple_kge_score'])
                        coef_differential = eval.mean_diffs_by_confidence(feature_df)
                        coef_means = eval.mean_by_confidence(feature_df, column='coefficient')
                        metrics.append({'experiment': experiment,
                                        'n_samples': n,
                                        'score_heuristic': ndcg_score(y_true=y_true,
                                                                      y_score=[feature_df['baseline_path_score']]),
                                        'linklogic': ndcg_score(y_true=y_true, y_score=[feature_df['coefficient']]),
                                        'linklogic_product': ndcg_score(y_true=y_true,
                                                                        y_score=[feature_df['coefficient_product']]),
                                        'random': ndcg_score(y_true=y_true, y_score=[feature_df['random']]),
                                        'coef_differential': coef_differential,
                                        'coef_non_benchmark': coef_means.get('0.0'),
                                        'coef_benchmark_1': coef_means.get('1.0'),
                                        'coef_benchmark_2': coef_means.get('2.0'),
                                        'coef_benchmark_3': coef_means.get('3.0'),
                                        'num_true': num_true_candidates,
                                        'query_triple': s_triple,
                                        'query_triple_kge_score': kge_score,
                                        'fidelity_train': fidelity_train,
                                        'fidelity': fidelity_test})

                        if plot_individual_results:
                            fig = px.scatter(feature_df, x='baseline_path_score', y='coefficient', color='label',
                                             hover_data=["path"], title=f'features: {s_triple}')
                            fig.show()

    all_features = pd.concat(feature_df_list, axis=0)

    #melt_stats = all_features.melt(id_vars=['bmk_category', 'experiment', 'n_samples', 'query_triple', 'path'])
    #melt_stats.head()

    #experiment_names = {'': 'Including child relation', '_fb12': 'FB12: single child relation removed'}
    #for experiment in experiments:
    #    X = melt_stats[melt_stats['experiment'] == experiment]
    fig = px.box(all_features, y='bmk_category', x='coefficient', facet_col='experiment', facet_row='n_samples',
                     hover_data=['query_triple'], title='Distribution of link logic coefficients across path types', height=900, width=900)
    fig.update_xaxes(matches=None)
    fig.show()

    M = pd.DataFrame(metrics)
    for exp, df in M.groupby('experiment'):
        print(exp)
        _ = df[['coef_non_benchmark', 'coef_benchmark_2', 'coef_benchmark_3', 'coef_differential']].hist()


    overlay = 'num_true'

    plot_scatter_with_facets(M=M, x='random', y='linklogic', color=overlay,
                             filename='LinkLogic_v_Random.html')

    plot_scatter_with_facets(M=M, x='score_heuristic', y='linklogic', color=overlay,
                             filename='LinkLogic_v_RawFeatureScores.html')

    plot_scatter_with_facets(M=M, x='random', y='score_heuristic', color=overlay,
                             filename='RawFeatureScores_v_Random.html')



def plot_scatter_with_facets(M, x, y, color, palette='matter', filename=None, size=8, width=900, height=300):
    fig = px.scatter(M, x=x, y=y, color=color, facet_col='n samples', facet_row='experiment',
                     hover_data=["query triple", "kge score"],
                     color_continuous_scale=palette, opacity=0.5, width=width, height=height)
    fig.add_shape(dict(type="line", x0=0, y0=0, x1=1.1, y1=1.1, line_width=1),
                  row="all", col="all",
                  line=dict(color="gray", width=1, dash="dot"))

    fig.update_traces(marker_size=size, marker_line={'width': 0.5, 'color': 'lightgrey'})

    fig.layout.plot_bgcolor = '#f0f0f0'
    fig.show()
    if filename is not None:
        fig.write_html(filename)



if __name__ == '__main__':
    main()
