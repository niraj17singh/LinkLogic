import numpy as np
import pandas as pd
from surrogate_model import *
from utils import parse_query, sort_entities, logsum1, logsum2

class LinkLogic(object):

    def __init__(self, query, graph, params, kge, e_emb, r_emb, e2id, rev_e2id, r2id, rev_r2id, e2type):
        self.kge = kge
        self.graph = graph
        self.query = query
        self.e_emb = e_emb
        self.r_emb = r_emb
        self.e2id = e2id
        self.r2id = r2id
        self.rev_r2id = rev_r2id
        self.rev_e2id = rev_e2id
        self.e2type = e2type

        # Linklogic Params
        self.prob = params["prob"] #Boolean for sigmoid transformation on the kge scores
        self.neighbor_sample_size = params["neighbor_sample_size"] #Number of neighbors to sample to calculate variance for perturbing query embeddings
        self.var_scale_head = params["var_scale_head"] #To scale the head embedding varinace for perturbation
        self.var_scale_tail = params["var_scale_tail"] #To scale the tail embedding variance for perturbation
        self.n_instances = params["n_instances"] # Number of perturbed queries to create
        self.seed = params["seed"] # To reproduce the results
        self.hop2_path_cal = params["hop2_path_cal"]
        self.topk = params["topk"] # Number of paths to consider per relation type
        self.logsum = params["logsum"] # Boolean for log transformation of featuers and the labels
        self.r1_name_list = params["r1_name_list"] # List of relations to consider for 1st hop in creating 2-hop paths
        self.r2_name_list = params["r2_name_list"] # List of relations to consider for 2nd hop in creating 2-hop paths
        self.consider_child = params["consider_child"] # Boolean to remove direct inverse evidence for parents benchmark
        self.alpha = params["alpha"] # Regularization constant to for the surrogate model
        self.feature_considerations = params["feature_considerations"] # Wheather to cosider only 1-hop, 2-hop or all features to train the surrogate model


    # Sample neighborhood
    def _sample_neighbors(self, e, r, neighbor_sample_size):
        '''
        Finds neighbors for a given entity and relation type
        :param e: entity id; for which we need to find neighbors
        :param r: relation id; neighbors are conditioned on a relation type
        :param neighbor_sample_size: Number of neighbors used
        :return: list of entities that are neighbors of "e"
        '''
        head_scores = self.kge.run(self.e_emb[[self.e2id[e]]], self.r_emb[[self.r2id[r]]], self.e_emb, prob=self.prob)
        preds = sort_entities(head_scores[0, 0, :], prob=self.prob)
        tail_scores = self.kge.run(self.e_emb, self.r_emb[[self.r2id[r]]], self.e_emb[preds[:neighbor_sample_size]],
                                   prob=self.prob)
        c = sort_entities(tail_scores, prob=self.prob)[:neighbor_sample_size, :, :]
        neighbors = list(set(c.flatten()))
        return neighbors

        # Find variance

    def _get_variance(self, e, neighbors, scale=1):
        '''
        Find the variance in the entity embeddings based on variance
        :param e: entity id;
        :param neighbors: neighbors of "e"
        :param scale: constant to scale the variance
        :return: variance; a scalar
        '''
        a = np.array([(abs(np.array(self.e_emb[idx]) - np.array(self.e_emb[self.e2id[e]]))) for idx in neighbors])
        b = np.mean(a, axis=1)
        c = b ** 2
        d = np.sum(c)
        var = d / len(neighbors)
        var = scale * var ** (0.5)
        return var

    # Perturb_embeddings
    def _perturb_embeddings(self, emb, n_instances, var, seed):
        '''
        Create perturbed embeddings for a given query emb
        :param emb: query embeddings to perturb
        :param n_instances: number of samples to create
        :param var: variance used for sampling normal distribution
        :param seed: to reproduce the results
        :return: an array with shape (n_instance, len(emb))
        '''
        e_hat = []
        np.random.seed(seed)
        for i in range(0, n_instances):
            noise = np.random.normal(0, var, len(emb))
            e_hat.append(list(emb + noise))
        return np.array(e_hat)

    def get_neighbors(self, query):
        '''
        Get the neighbors for the query triple
        :param query: a tuple of (hq, rq, tq)
        :return: a tuple (list of neighbors of hq, list of neighbors of tq)
        '''
        hq, rq, tq = parse_query(query)
        head_neighbors = self._sample_neighbors(hq, rq, self.neighbor_sample_size)
        tail_neighbors = self._sample_neighbors(tq, rq, self.neighbor_sample_size)
        return head_neighbors, tail_neighbors

    def get_var(self, query, head_neighbors, tail_neighbors):
        '''

        :param query:
        :param head_neighbors:
        :param tail_neighbors:
        :return:
        '''
        hq, rq, tq = parse_query(query)
        head_var = self._get_variance(hq, head_neighbors, scale=self.var_scale_head)
        tail_var = self._get_variance(tq, tail_neighbors, scale=self.var_scale_tail)
        return head_var, tail_var

    def get_query_perturbings(self, query, head_var, tail_var):
        hq, rq, tq = parse_query(query)
        head_pert = self._perturb_embeddings(self.e_emb[self.e2id[hq]], self.n_instances, head_var,
                                             self.seed)
        tail_pert = self._perturb_embeddings(self.e_emb[self.e2id[tq]], self.n_instances, tail_var,
                                             self.seed)
        return head_pert, tail_pert

    def _hop1_score(self, e, inverse=None):
        if inverse == False:
            scores = self.kge.run(self.e_emb[[self.e2id[e]]], self.r_emb, self.e_emb, prob=self.prob)
        else:
            scores = self.kge.run(self.e_emb, self.r_emb, self.e_emb[[self.e2id[e]]], prob=self.prob)
        return scores

    def get_1hop_links(self, query):
        hq, rq, tq = parse_query(query)

        # head features
        head_scores = self._hop1_score(hq, inverse=False)
        head_inv_scores = self._hop1_score(hq, inverse=True)

        # tail features
        tail_scores = self._hop1_score(tq, inverse=False)
        tail_inv_scores = self._hop1_score(tq, inverse=True)
        return head_scores, head_inv_scores, tail_scores, tail_inv_scores

    def get_2hop_links(self, head_scores, tail_scores, head_inv_scores, tail_inv_scores):

        intermediate_direct_scores = np.zeros([1, len(self.r2id), len(self.e2id), len(self.r2id), 1])
        intermediate_inv_scores = np.zeros([1, len(self.r2id), len(self.e2id), len(self.r2id), 1])

        for i in range(len(self.r2id)):
            for j in range(len(self.r2id)):
                if self.hop2_path_cal == "product":
                    intermediate_direct_scores[0, i, :, j, 0] = head_scores[0, i, :] * tail_inv_scores[:, j, 0]
                    intermediate_inv_scores[0, i, :, j, 0] = tail_scores[0, i, :] * head_inv_scores[:, j, 0]
                elif self.hop2_path_cal == "sqrt":
                    intermediate_direct_scores[0, i, :, j, 0] = np.sqrt(head_scores[0, i, :] * tail_inv_scores[:, j, 0])
                    intermediate_inv_scores[0, i, :, j, 0] = np.sqrt(tail_scores[0, i, :] * head_inv_scores[:, j, 0])
        return intermediate_direct_scores, intermediate_inv_scores

    def _1hop_features(self, e, hop1_scores, pert_emb, rel_list, inverse=None):

        hop1_feature = []
        head_feature_name = []

        if not inverse:
            for r in rel_list:
                preds = sort_entities(hop1_scores[0, r, :], prob=self.prob)[:self.topk]
                head_feature_name.extend([[e, self.rev_r2id[r], self.rev_e2id[pred]] for pred in list(preds)])
                score = self.kge.run(pert_emb, self.r_emb[[r]], self.e_emb[preds], prob=self.prob)
                hop1_feature.append(score)

            hop1_feature = np.array(hop1_feature).transpose((1, 0, 3, 2))
            hop1_feature = hop1_feature.reshape(self.n_instances, len(rel_list) * self.topk)

        else:
            for r in rel_list:
                preds = sort_entities(hop1_scores[:, r, 0], prob=self.prob)[:self.topk]
                hop1_feature.extend([[self.rev_e2id[pred], self.rev_r2id[r], e] for pred in list(preds)])
                score = self.kge.run(self.e_emb[preds], self.r_emb[[r]], pert_emb, prob=self.prob)
                hop1_feature.append(score)

            hop1_feature = np.array(hop1_feature).transpose((3, 0, 2, 1))
            hop1_feature = hop1_feature.reshape(self.n_instances, len(rel_list) * self.topk)

        if self.logsum:
            hop1_feature = logsum1(hop1_feature)

        return hop1_feature, head_feature_name

    def get_1hop_features(self, query, head_pert, tail_pert):
        hq, _, tq = parse_query(query)

        print("Getting head features")
        head_scores, head_inv_scores, tail_scores, tail_inv_scores = self.get_1hop_links(query)

        rel_list = list(self.rev_r2id.keys())

        # Get head features
        head_features, head_features_name = self._1hop_features(hq, head_scores, head_pert, rel_list, inverse=False)

        # Get tail features
        tail_features, tail_features_name = self._1hop_features(tq, tail_scores, tail_pert, rel_list, inverse=False)



        return head_scores, head_inv_scores, tail_scores, tail_inv_scores, head_features, \
               head_features_name, tail_features, tail_features_name

    def _2hop_features(self, h, t, intermediate_score, head_pert, tail_pert):

        r1_list = [self.r2id[r] for r in self.r1_name_list]
        r2_list = [self.r2id[r] for r in self.r2_name_list]

        intermediate_feature = []
        intermediate_feature_name = []
        for r1 in r1_list:
            for r2 in r2_list:
                preds = sort_entities(intermediate_score[0, r1, :, r2, 0], prob=self.prob)[:self.topk]
                intermediate_feature_name.extend(
                    [[h, self.rev_r2id[r1], self.rev_e2id[pred], self.rev_r2id[r2], t] for pred in list(preds)])

                score1 = self.kge.run(head_pert, self.r_emb[[r1]], self.e_emb[preds], prob=self.prob)
                score2 = self.kge.run(self.e_emb[preds], self.r_emb[[r2]], tail_pert, prob=self.prob)

                score2 = score2.transpose((2, 1, 0))
                if self.logsum:
                    intermediate_feature.append(logsum2(score1, score2))
                elif self.hop2_path_cal == "product" and not self.logsum:
                    intermediate_feature.append(score1 * score2)
                elif self.hop2_path_cal == "sqrt" and not self.logsum:
                    intermediate_feature.append(np.sqrt(score1 * score2))

        intermediate_feature = np.array(intermediate_feature).transpose((1, 0, 3, 2))
        intermediate_feature = intermediate_feature.reshape(self.n_instances,
                                                            len(r1_list) * len(r2_list) * self.topk)

        return intermediate_feature, intermediate_feature_name

    def get_2hop_features(self,
                          query,
                          head_scores,
                          tail_scores,
                          head_inv_scores,
                          tail_inv_scores,
                          head_pert,
                          tail_pert):

        hq, rq, tq = parse_query(query)

        intermediate_direct_scores, intermediate_inv_scores = self.get_2hop_links(head_scores, tail_scores,
                                                                                  head_inv_scores, tail_inv_scores)

        intermediate_direct_feature, intermediate_direct_feature_name = self._2hop_features(hq, tq,
                                                                                            intermediate_direct_scores,
                                                                                            head_pert, tail_pert)
        intermediate_inv_feature, intermediate_inv_feature_name = self._2hop_features(tq, hq,
                                                                                      intermediate_inv_scores,
                                                                                      tail_pert, head_pert)

        return intermediate_direct_feature, \
               intermediate_direct_feature_name, \
               intermediate_inv_feature, \
               intermediate_inv_feature_name

    def get_labels(self, query, head_pert, tail_pert):
        hq, rq, tq = parse_query(query)
        labels = []
        for i in range(self.n_instances):
            score = self.kge.run(head_pert[[i]], self.r_emb[[self.r2id[rq]]], tail_pert[[i]], prob=self.prob)
            if self.logsum:
                score = logsum1(score)
            labels.append(score.flatten()[0])
        return labels

    def _select_features(self, head_features,
                         head_feature_name, tail_features, tail_feature_name,
                         intermediate_direct_features, intermediate_direct_feature_name,
                         intermediate_inv_features, intermediate_inv_feature_name,
                         feature_consideration = "all"):
        columns = []
        if feature_consideration == "all":
            data_arr = np.concatenate(
                (head_features, tail_features, intermediate_direct_features, intermediate_inv_features), axis=1)
            columns = head_feature_name + tail_feature_name + intermediate_direct_feature_name + intermediate_inv_feature_name

        elif feature_consideration == "2hop":
            data_arr = np.concatenate((intermediate_direct_features, intermediate_inv_features), axis=1)
            columns = intermediate_direct_feature_name + intermediate_inv_feature_name

        elif feature_consideration == "1hop":
            data_arr = np.concatenate((head_features, tail_features), axis=1)
            columns = head_feature_name + tail_feature_name

        features = pd.DataFrame(data_arr)
        return features, columns

    def _remove_redundant_features(self, query, data, columns):
        hq, rq, tq = parse_query(query)
        # Remove redundant features
        drop_features_idx = set()
        # Filter based on Entity types
        for i, c in enumerate(columns):  # Remove 1hop paths with inconsistent entity type and relations
            if c[1] in ["children", "spouse", "parents"] and self.e2type[c[2]] != "Person":
                drop_features_idx.add(i)
            elif c[1] in ["location", "place_of_birth", "place_of_death", "nationality"] and self.e2type[c[2]] != "Location":
                drop_features_idx.add(i)
            elif c[1] in ["gender"] and self.e2type[c[2]] != "Gender":
                drop_features_idx.add(i)
            elif c[1] in ["profession"] and self.e2type[c[2]] != "Profession":
                drop_features_idx.add(i)
            elif c[1] in ["institution"] and self.e2type[c[2]] != "Institution":
                drop_features_idx.add(i)
            elif c[1] in ["ethnicity"] and self.e2type[c[2]] != "Ethnicity":
                drop_features_idx.add(i)
            elif c[1] in ["cause_of_death"] and self.e2type[c[2]] != "CauseOfDeath":
                drop_features_idx.add(i)
            elif c[1] in ["religion"] and self.e2type[c[2]] != "Religion":
                drop_features_idx.add(i)

        # Filter based on redundant features
        for i, c in enumerate(columns):
            if c[0] == c[2]:  # Remove 1hop paths with same entity type on head and tail
                drop_features_idx.add(i)

        for i, c in enumerate(columns):
            if c[0] == hq and c[1] == rq and c[2] == tq:  # Remove query triple
                drop_features_idx.add(i)

        for i, c in enumerate(columns):
            if len(c) == 5:  # Remove 2hop paths with inconsistent entity type and relations
                if c[3] in ["children", "spouse", "parents"] and self.e2type[c[4]] != "Person":
                    drop_features_idx.add(i)
                elif c[3] in ["location", "place_of_birth", "place_of_death", "nationality"] and self.e2type[
                    c[4]] != "Location":
                    drop_features_idx.add(i)
                elif c[3] in ["gender"] and self.e2type[c[4]] != "Gender":
                    drop_features_idx.add(i)
                elif c[3] in ["profession"] and self.e2type[c[4]] != "Profession":
                    drop_features_idx.add(i)
                elif c[3] in ["institution"] and self.e2type[c[4]] != "Institution":
                    drop_features_idx.add(i)
                elif c[3] in ["ethnicity"] and self.e2type[c[4]] != "Ethnicity":
                    drop_features_idx.add(i)
                elif c[3] in ["cause_of_death"] and self.e2type[c[4]] != "CauseOfDeath":
                    drop_features_idx.add(i)
                elif c[3] in ["religion"] and self.e2type[c[4]] != "Religion":
                    drop_features_idx.add(i)

        for i, c in enumerate(columns):
            if len(c) == 3:
                if self.e2type[c[0]] != "Person":  # Remove 1hop paths with inconsistent entity type and relations
                    drop_features_idx.add(i)
            if len(c) == 5:  # Remove 2hop paths with inconsistent entity type and relations
                if self.e2type[c[2]] != "Person":
                    drop_features_idx.add(i)

        for i, c in enumerate(columns):
            if len(c) == 5:
                if c[2] == c[4]:  # Remove 2hop paths that have same entities
                    drop_features_idx.add(i)
                if c[1] == c[3]:  # Remove 2hop paths that have same relations
                    drop_features_idx.add(i)

        # Remove direct children relationship
        if not self.consider_child:
            for i, c in enumerate(columns):
                if len(c) == 3:
                    if c[0] == tq and c[1] == "children" and c[2] == hq:
                        drop_features_idx.add(i)

        seen = []
        for i, c in enumerate(columns):
            if c in seen:
                drop_features_idx.add(i)
            else:
                seen.append(c)

        print("\nTotal redundant features removed: ", len(drop_features_idx))

        final_columns = []
        for i, x in enumerate(columns):
            if i not in drop_features_idx:
                final_columns.append(x)

        data = data.drop(drop_features_idx, axis=1)

        return data, final_columns

    def train_surrogate_model(self, data, columns):
        sm = SurrogateModel(model="Lasso", data=data, alpha=self.alpha)
        train_acc, test_acc, coef, train_x, test_x, train_y, test_y, pred_train, pred_test = sm.run()
        feature_stats = sm.append_stats_to_features(pd.DataFrame({"Features": columns}), coef)
        final_df = feature_stats.sort_values("Coefficient", ascending=False)
        return train_acc, test_acc, coef, train_x, test_x, train_y, test_y, pred_train, pred_test, final_df

    def store_model(self, query, final_df, columns, train_acc, test_acc):
        hq, rq, tq = parse_query(query)
        paths = final_df["Features"].tolist()
        coef = final_df["Coefficient"].tolist()

        kge_score = []
        split = []
        for p in final_df["Features"].tolist():
            if len(p) == 3:
                h, r, t = p[0], p[1], p[2]
                score = self.kge.run(self.e_emb[[self.e2id[h]]], self.r_emb[[self.r2id[r]]], self.e_emb[[self.e2id[t]]],
                                     prob=self.prob)
                if self.logsum:
                    score1 = logsum1(score)
                    path_score_method = "log_sum"
                    kge_score.append({"1st_hop_kge_score": list(score.flatten())[0],
                                      "path_score": list(score1.flatten())[0],
                                      "path_score_method": path_score_method})
                else:
                    path_score_method = "None"
                    kge_score.append({"1st_hop_kge_score": list(score.flatten())[0],
                                      "path_score": list(score.flatten())[0],
                                      "path_score_method": path_score_method})

                if (h, r, t) not in self.graph:
                    split.append(["None"])
                else:
                    split.append([self.graph[h, r, t]])

            elif len(p) == 5:
                h, r1, e, r2, t = p[0], p[1], p[2], p[3], p[4]
                score1 = self.kge.run(self.e_emb[[self.e2id[h]]],
                                      self.r_emb[[self.r2id[r1]]],
                                      self.e_emb[[self.e2id[e]]],
                                      prob=self.prob)
                score2 = self.kge.run(self.e_emb[[self.e2id[e]]],
                                      self.r_emb[[self.r2id[r2]]],
                                      self.e_emb[[self.e2id[t]]],
                                      prob=self.prob)
                #             score = score1 * score2abs
                if self.logsum:
                    score = logsum2(score1, score2)
                    path_score_method = "log_sum"

                elif self.hop2_path_cal == "product" and not self.logsum:
                    score = score1 * score2
                    path_score_method = "product"

                elif self.hop2_path_cal == "sqrt" and not self.logsum:
                    path_score_method = "sqrt"
                    score = np.sqrt(score1 * score2)

                kge_score.append({"1st_hop_kge_score": list(score1.flatten())[0],
                                  "2nd_hop_kge_score": list(score2.flatten())[0],
                                  "path_score": list(score.flatten())[0],
                                  "path_score_method": path_score_method})
                if (h, r1, e) not in self.graph:
                    split1 = "None"
                else:
                    split1 = self.graph[h, r1, e]

                if (e, r2, t) not in self.graph:
                    split2 = "None"
                else:
                    split2 = self.graph[e, r2, t]

                split.append([split1, split2])

        paths_final = []
        features = []
        for i, path in enumerate(paths):
            temp = {}
            temp["path"] = path
            temp["coef"] = coef[i]
            temp["kge_score"] = kge_score[i]
            temp["split"] = split[i]
            paths_final.append(temp)
            if coef[i] != 0:
                features.append(temp)

        triple_score = self.kge.run(self.e_emb[[self.e2id[hq]]],
                                    self.r_emb[[self.r2id[rq]]],
                                    self.e_emb[[self.e2id[tq]]],
                                    prob=self.prob)
        triple_score = list(triple_score.flatten())[0]

        dump = {}
        dump["query_triple"] = [hq, rq, tq]
        dump["prob"] = self.prob
        dump["query_triple_kge_score"] = str(triple_score)
        dump["linklogic_explanations"] = features
        dump["linklogic_features"] = paths_final
        dump["final_columns"] = columns
        dump["linklogic_metrics"] = {"train_acc": train_acc, "test_acc": test_acc}
        return dump

    def run(self, query):
        print("---"*50)
        print("\nQuery:", query)
        hq, rq, tq = parse_query(query)
        
        

        head_neighbors, tail_neighbors = self.get_neighbors(query)
        head_var, tail_var = self.get_var(query, head_neighbors, tail_neighbors)
        head_pert, tail_pert = self.get_query_perturbings(query, head_var, tail_var)

        print("\nGetting features...")
        print("\n1. 1-hop features...")
        head_scores, head_inv_scores, tail_scores, tail_inv_scores, head_features, \
        head_feature_name, tail_features, tail_feature_name = self.get_1hop_features(query, head_pert, tail_pert)

        print("\n2. 2-hop features...")
        intermediate_direct_features, intermediate_direct_feature_name, \
        intermediate_inv_features, intermediate_inv_feature_name = self.get_2hop_features(query,
                                                                                         head_scores,
                                                                                         tail_scores,
                                                                                         head_inv_scores,
                                                                                         tail_inv_scores,
                                                                                         head_pert,
                                                                                         tail_pert)
        print("\nGetting Labels...")
        labels = self.get_labels(query, head_pert, tail_pert)


        summary = {}
        for feature_consideration in self.feature_considerations:
            print("\nConsidering {} features".format(feature_consideration))
            features, columns = self._select_features(head_features,
                             head_feature_name, tail_features, tail_feature_name,
                             intermediate_direct_features, intermediate_direct_feature_name,
                             intermediate_inv_features, intermediate_inv_feature_name,
                             feature_consideration=feature_consideration)
            features["Labels"] = labels
            data = features
            print("\nTotal features identified: ", data.shape[1])
            print("\nRemoving redundant features...")
            data, columns  = self._remove_redundant_features(query, data, columns)

            print("\nTraining surrogate model")
            train_acc, \
            test_acc, \
            coef, \
            train_x, test_x, train_y, test_y, pred_train, pred_test, final_df = self.train_surrogate_model(data, columns)
            
            print("\nDumping results")
            dump = self.store_model(query, final_df, columns, train_acc, test_acc)
            summary[feature_consideration] = dump
        print("---"*50)
        return summary
