import numpy as np
import torch

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

class KGE():
    def __init__(self, method="TransE"):
        if method == "TransE":
            self.run = self.TransE
        elif method == "ComplEx":
            self.run = self.ComplEx
           
    def TransE(self, e_h, e_r, e_t, prob=False):
        e_h = np.expand_dims(e_h, axis=(1, 2))
        e_r = np.expand_dims(e_r, axis=(0, -2))
        e_t = np.expand_dims(e_t, axis=(0, 1))
    #     print(e_h.shape)
    #     print(e_r.shape)
    #     print(e_t.shape)
        score = e_h + e_r - e_t
    #     print(score.shape)
        score = np.linalg.norm(score, axis=-1)
        if prob:
            score = 1 - sigmoid(score)
        return score
    
    def ComplEx(self ,e_h, e_r, e_t, prob=False):
        r"""Forward pass of ComplEx decoder.

        Args:
            head (Tensor): head entities of shape (batch_size, 1, embedding_dim)
            relation (Tensor): relation of shape (batch_size, 1, embedding_dim)
            tail (Tensor): tail entities of shape (batch_size, num_targets, embedding_dim)
            answer_side (str): Whether the query is head or tail (used for broadcasting)

        Returns:
            score (batch_size, num_targets)
        """

        e_h = np.expand_dims(e_h, axis=(1, 2))
        e_t = np.expand_dims(e_t, axis=(0, 1))
        e_r = np.expand_dims(e_r, axis=(0, 2 ))

        e_h = torch.from_numpy(e_h)
        e_t = torch.from_numpy(e_t)
        e_r = torch.from_numpy(e_r)


        re_head, im_head = torch.chunk(e_h, 2, dim=-1)
        re_relation, im_relation = torch.chunk(e_r, 2, dim=-1)
        re_tail, im_tail = torch.chunk(e_t, 2, dim=-1)

        score = re_head * re_relation * re_tail - im_head * im_relation * re_tail + re_head * im_relation * im_tail + im_head * re_relation * im_tail
        score = score.sum(dim=-1)
#         print(score.shape)
        if prob:
            score = sigmoid(score)
        return np.array(score)
