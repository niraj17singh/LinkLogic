import argparse
import logging
import json
import os
import pandas as pd
import pickle
from tqdm import tqdm

from kge import *
from utils import *
from linklogic import LinkLogic
from surrogate_model import *

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Run LinkLogic')
    parser.add_argument('--params_file', default="tmp", type=str, help='should be a json file with all the parameters for the model')
    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args()

    with open(args.params_file) as f:
        params = json.load(f)

    io_params = params["io_params"]
    linklogic_params = params["linklogic_params"]

    ### Read the data

    # Read entity mapping
    e2id_path = open(io_params["data_path"] + "/" + linklogic_params["dataset"] + "/entity2id.txt", "r")
    e2id = {}
    for line in e2id_path:
        line = line.strip()
        e, ids = line.split("\t")
        e2id[e] = int(ids)
    rev_e2id = {v: k for k, v in e2id.items()}

    # Read relation mapping
    r2id_path = open(io_params["data_path"] + "/" + linklogic_params["dataset"] + "/relation2id.txt", "r")
    r2id = {}
    for line in r2id_path:
        line = line.strip()
        r, ids = line.split("\t")
        r2id[r] = int(ids)
    rev_r2id = {v: k for k, v in r2id.items()}

    e2type_path = io_params["data_path"] + "/" + linklogic_params["dataset"] + "/entity_names_to_types.csv"
    e2type_raw_data = pd.read_csv(e2type_path)
    e2type = {}
    for row in e2type_raw_data.iterrows():
        e2type[row[1][0]] = row[1][1]

    train_path = io_params["data_path"] + "/" + linklogic_params["dataset"] + "/train.txt"
    test_path = io_params["data_path"] + "/" + linklogic_params["dataset"] + "/test.txt"
    valid_path = io_params["data_path"] + "/" + linklogic_params["dataset"] + "/valid.txt"

    train_path = open(train_path)
    test_path = open(test_path)
    valid_path = open(valid_path)

    # Read all the triples from the train, test, and valid dataset to have the fully informed data
    graph = {}
    for t in train_path:
        t = t.strip()
        e1, r, e2 = t.split("\t")
        graph[e1, r, e2] = "train"

    for t in test_path:
        t = t.strip()
        e1, r, e2 = t.split("\t")
        graph[e1, r, e2] = "test"

    for t in valid_path:
        t = t.strip()
        e1, r, e2 = t.split("\t")
        graph[e1, r, e2] = "valid"

    # For parents benchmark dataset
    tuning_path = io_params["data_path"] + "/" + linklogic_params[
        "dataset"] + "/commonsense_benchmark_for_tuning_v2.json"
    analysis_path = io_params["data_path"] + "/" + linklogic_params[
        "dataset"] + "/commonsense_benchmark_for_analysis_v2.json"
    all_path = io_params["data_path"] + "/" + linklogic_params["dataset"] + "/commonsense_benchmark_all_v2.json"


    # Read benchmark dataset
    data = {}
    with open(tuning_path) as f:
        data["tuning"] = json.load(f)

    with open(analysis_path) as f:
        data["analysis"] = json.load(f)

    with open(all_path) as f:
        data["all"] = json.load(f)

    category = linklogic_params["benchmark"]
    bmk = {}
    for t in linklogic_params["benchmark_datatype"]:
        bmk[t] = {}
        bmk[t]["train"] = []
        bmk[t]["test"] = []
        for d in data[t]:
            if d['category'] == category:
                if d["split"] == "train":
                    bmk[t]["train"].append(d["query_triple"])
                elif d["split"] == "test":
                    bmk[t]["test"].append(d["query_triple"])

    # Read the trained embeddings
    method = linklogic_params["method"]
    e_emb = np.load(
        io_params["data_path"] + "/" + linklogic_params["dataset"] + f"/embeddings/{method}/entity_embedding.npy")
    r_emb = np.load(
        io_params["data_path"] + "/" + linklogic_params["dataset"] + f"/embeddings/{method}/relation_embedding.npy")

    # KGE object used to get the kge scores
    kge = KGE(method=linklogic_params["method"])

    # Generating results for akbc paper
    for dataset in linklogic_params["benchmark_datatype"]:
        summary_list = {}
        summary_list["all"] = []
        summary_list["1hop"] = []
        summary_list["2hop"] = []
        for split in ["train", "test"]:
            for t in tqdm(bmk[dataset][split]):
                query = t[0], t[1], t[2]
                LL = LinkLogic(query, graph, linklogic_params, kge, e_emb, r_emb, e2id, rev_e2id, r2id, rev_r2id,
                               e2type)
                summary = LL.run(query)

                for feature_consideration in linklogic_params["feature_considerations"]:
                    summary[feature_consideration]["category"] = category
                    summary[feature_consideration]["split"] = split
                    summary[feature_consideration]["linklogic_params"] = linklogic_params
                    summary_list[feature_consideration].append(summary[feature_consideration])
        for feature_consideration in linklogic_params["feature_considerations"]:
            if not os.path.exists(io_params["save_path"]):
                print("Creating save path: ", io_params["save_path"])
                os.makedirs(io_params["save_path"])
            with open(io_params["save_path"] + "/" +  linklogic_params["dataset"] + f'_{category}_{dataset}_{method}_{feature_consideration}_child_{linklogic_params["consider_child"]}.pickle',
                    'wb') as handle:
                pickle.dump(summary_list[feature_consideration], handle, protocol=pickle.HIGHEST_PROTOCOL)
