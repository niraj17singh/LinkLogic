# LinkLogic

This is the code repository for the paper titled LinkLogic: A New Method and Benchmark for Explainable
Knowledge Graph Predictions.

# Datasets
- The `FB13` dataset from OpenKE benchmark is present at `dataset/FB13`
- The `dataset/fb13_resplit` resplits the test data in `FB13` dataset for evaluation purposes mentioned in the paper.
- The new `FB14` dataset is present at `dataset/fb14`


#  Compute Knowledge Graph Embeddings

To train the CompleX Decoder used in the paper, we used the [DGL-KE](https://github.com/awslabs/dgl-ke) library. For easy reproducibility of the experiments mentioned in the paper, we are sharing the trained embeddings using ComplEx decode for the `fb13_resplit` and `fb14` datasets. To download the same run the following script from the root of the project:

```bash
sh get_embeddings.sh
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
    - Description of the `linklogic_params`
        - dataset: 
        - method: 
        - prob: Boolean for sigmoid transformation on the knowledge graph embedding scores
        - n_instances: Number of neighbors to sample to calculate variance for perturbing query embeddings
        - topk: Number of paths to consider per relation type
        - neighbor_sample_size: Number of neighbors to sample to calculate variance for perturbing query embeddings
        - var_scale_head: To scale the head embedding varinace for perturbation
        - var_scale_tail: To scale the tail embedding variance for perturbation
        - seed: To reproduce the results
        - hop2_path_cal: 
        - logsum: Boolean for log transformation of featuers and the labels
        - alpha: Regularization constant to for the surrogate model
        - consider_child:  
        - benchmark
        - benchmark_datatype:
        - r1_name_link:
        - r2_name_list
        - feature_considerations: 

    
3. Run Linklogic
    ```bash
    cd linklogic
    python run_linklogic --params_file params.json
    ```
4. Generated results are stored in the `io_path["save_path"]`


# Citation

If you use  LinkLogic in a scientific publication, we would appreciate citations to the following paper:

: TODO
```bibtex
@article{xyz,
  title={LinkLogic: A New Method and Benchmark for Explainable Knowledge Graph Predictions},
  author={Niraj Kumar Singh, Gustavo Polleti, Saee Paliwal, Rachel Hodos Nkhereanye},
  journal={},
  year={2024}
}
```

# License
This project is licensed under the MIT License.




