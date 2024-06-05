# LinkLogic

This is the code repository for the paper titled LinkLogic: A New Method and Benchmark for Explainable
Knowledge Graph Predictions.

# Datasets
- The `FB13` dataset from OpenKE benchmark is present at `dataset/FB13`
- The `dataset/fb13_resplit` resplits the test data in `FB13` dataset for evaluation purposes mentioned in the paper.
- The new `FB14` dataset is present at `dataset/fb14`


#  Compute Knowledge Graph Embeddings

To train the ComplEx Decoder used in the paper, we used the [DGL-KE](https://github.com/awslabs/dgl-ke) library. For easy reproducibility of the experiments mentioned in the paper, we are sharing the trained embeddings using ComplEx decode for the `fb13_resplit` and `fb14` datasets. To download the same run the following script from the root of the project:

```bash
brew install git-lfs
git lfs pull
```

# LinkLogic Explanations

1. Setup environment
    ```bash
    conda create -n linklogic python=3.8
    conda activate linklogic
    ```
2. Update Parameters 
    - To run the experiemnts configure the params at `linklogic/params.json`
    - It contains two set of params
        - `io_params`: Defines the params to read the dataset and save the outputs
        - `linklogic_params`: Defines the params used by linklogic to generate explanations
    - Description of the `io_params`
        - `data_path`: Path to the dataset.
        - `save_path`: Path to save the output of the `run_linklogic.py`

    - Description of the `linklogic_params`
        - `dataset`: Name of the dataset. Currently supports `fb13_resplit`, `fb14`
        - `method`: Name of the KGE embedding strategy. Currently supported `ComplEx` and `TransE`.
        - `prob`: Boolean for sigmoid transformation on the knowledge graph embedding scores
        - `n_instances`: Number of neighbors to sample to calculate variance for perturbing query embeddings
        - `topk`: Number of paths to consider per relation type
        - `neighbor_sample_size`: Number of neighbors to sample to calculate variance for perturbing query embeddings
        - `var_scale_head`: To scale the head embedding varinace for perturbation. Defaul value = 1
        - `var_scale_tail`: To scale the tail embedding variance for perturbation. Defaul value = 1
        - `seed`: To reproduce the results
        - `hop2_path_cal`: Strategy to compute the 2-hop path score. Valid options - `product` or `sqrt`
        - `logsum`: Boolean for log transformation of featuers and the labels
        - `alpha`: Regularization constant to for the surrogate model
        - `consider_child`: Boolean to remove direct inverse evidence for parents benchmark
        - `benchmark`: Benchmark category - Currently supports `parents` or `location`
        - `benchmark_datatype`: Benchmark datatype - Currently supports `analysis` or `tuning`
        - `r1_name_list`: List of relations to consider for 1st hop in creating 2-hop paths
        - `r2_name_list`: List of relations to consider for 2nd hop in creating 2-hop paths
        - `feature_considerations`: Wheather to cosider only 1-hop, 2-hop or all features to train the surrogate model

    
3. Run Linklogic
    ```bash
    cd linklogic
    python run_linklogic --params_file params.json
    ```
4. Generated results are stored in the `io_path["save_path"]`

5. Format of generated results
    - For each `run_linklogic.py` run results are saved in the `io_params["save_path"]` with the following file format:
        - {dataset}\_{benchmark}\_{benchmark_datatype}\_{method}\_{feature_considerations}\_child\_{consider\_child}.pickle
        - Here, for each <feature_consideration> a new .pickle file is saved. 
    - Contents of the .pickle files saved:
        - Each file constains the metadata associated with the list of all the queries
        - Output Params:
            - `query_triple`: The query triple for which the linklogic explanations are desired. 
            - `prob`: Boolean for sigmoid transformation on the knowledge graph embedding scores
            - `query_triple_kge_score`: The link prediction score as identified by the KGE method used. Default method is ComplEx and the scores are between 0 and 1 if `prob` is `True`.
            - `final_columns`: List of all the triples identified as a feature for 1-hop, 2-hop and all. 
            - `linklogic_features`: List of all the triples identified as a feature for 1-hop, 2-hop and all, along with other metadata. Eg. `kge_score` which is a dictionary of `1st_hop_kge_score`, `path_score` and `path_score_method`, `coef` which us the coefficient score from the surrogate model, and `split` which contains the information if the triple is present in the `train`, `valid` or `test` split in the graph.
            - `linklogic_metrics`: Dictionary that contains two keys: 1. `train_acc` - Training accurary of the surrogate model, and 2. `test_acc` - Test accuracy of the surrogate model. 
            - `category`: Benchmark category - Currently supports `parents` or `location`
            - `split`: Information on whether the query triple belongs to `train`, `test` or `valid` split.
            - `linklogic_explanations`: A subset of `linklogic_features` which are linklogic explanations based on the positive coeficient score from the surrogate model. 
            - `linklogic_params`: A copy of the `params.json` file used to generate the explanations for reproducibility.

# Citation

If you use  LinkLogic in a scientific publication, we would appreciate citations to the following paper:

```bibtex
@article{kumar-singh2024linklogic,
  title={LinkLogic: A New Method and Benchmark for Explainable Knowledge Graph Predictions},
  author={Kumar-Singh, Niraj and Polleti, Gustavo and Paliwal, Saee and Hodos-Nkhereanye, Rachel},
  journal={arXiv preprint arXiv:2406.00855},
  year={2024}
}
```

# License
This project is licensed under the MIT License.




