# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, field
import torch
from torch import nn
from torch.nn import functional as F

from params import Params


@dataclass
class PointerNetworkArgs(Params):
    enc_output: bool = field(
        default=True,
        metadata={
            "help": "Feed the pointer network with the encoder output (input otherwise)"
        },
    )
    enc_output_in_dec_input: bool = field(
        default=False,
        metadata={
            "help": "Feed the decoder with embeddings coming from encoder output (input otherwise)"
        },
    )
    proj_x: bool = field(
        default=True, metadata={"help": "Project source embeddings in pointer network"},
    )
    proj_y: bool = field(
        default=True, metadata={"help": "Project target embeddings in pointer network"},
    )
    proj_s: bool = field(
        default=True,
        metadata={"help": "Project scores in pointer network (instead of dot product)"},
    )


class PointerNetwork(nn.Module):
    """
    Pointer network module.
    https://arxiv.org/pdf/1506.03134.pdf
    https://arxiv.org/pdf/1609.07843.pdf
    """

    def __init__(self, dim, proj_x, proj_y, proj_s):
        super().__init__()
        self.dim = dim
        self.proj_x = nn.Linear(dim, dim) if proj_x else None
        self.proj_y = nn.Linear(dim, dim) if proj_y else None
        self.proj_s = nn.Linear(dim, 1, bias=False) if proj_s else None

    def forward(
        self, graph_sizes, embeddings, predicted_embs, target_ptr_ids, graph_ids
    ):
        """
        Compute pointer network scores.
        Input:
            - graph_sizes LongTensor(n_graphs,)
                n_graphs = number of considered graphs
            - embeddings FloatTensor(n_graphs, max_nodes, dim)
                max_nodes = maximum number of nodes in a graph
                (padded) embeddings of all nodes in each graph
            - predicted_embs FloatTensor(n_pointers, dim)
                n_pointers = number of pointers to predict
            - target_ptr_ids LongTensor(n_pointers,)
                local indices of pointers to predict, with values in [0, max_nodes - 1]
            - graph_ids LongTensor(n_pointers,)
                which graph each pointer belongs to
        Output:
            - scores.size() == (n_pointers, max_nodes)
                padded tensor of node scores
        """
        # input check
        n_graphs, max_nodes, dim = embeddings.size()
        n_pointers = predicted_embs.size(0)
        assert dim == self.dim
        assert graph_sizes.size() == (n_graphs,) and graph_sizes.max() == max_nodes
        assert predicted_embs.size() == (n_pointers, dim)
        assert target_ptr_ids.size() == (n_pointers,)
        assert graph_ids.size() == (n_pointers,)
        assert n_pointers == 0 or graph_ids.max() < n_graphs

        # Small hack to handle PyTorch distributed. Optimization will fail
        # if the module parameters are not used in the forward path.
        if n_pointers == 0:
            return 0 * sum(p.sum() for p in self.parameters())

        # selected respective graph embeddings
        graph_embs = embeddings[graph_ids]
        assert graph_embs.size() == (n_pointers, max_nodes, dim)

        # project embeddings
        x = graph_embs if self.proj_x is None else self.proj_x(graph_embs)
        y = predicted_embs if self.proj_y is None else self.proj_y(predicted_embs)
        assert x.size() == (n_pointers, max_nodes, dim)
        assert y.size() == (n_pointers, dim)

        # compute a score for all graph nodes
        if self.proj_s is None:
            scores = (x * y.unsqueeze(1)).sum(2)
        else:
            scores = self.proj_s(F.relu(x + y.unsqueeze(1))).squeeze(2)
        assert scores.size() == (n_pointers, max_nodes)

        # number of nodes per graph
        n_nodes_per_graph = graph_sizes[graph_ids]
        assert n_nodes_per_graph.size() == (n_pointers,)
        assert n_nodes_per_graph.min() >= 1
        assert all(target_ptr_ids < n_nodes_per_graph)

        # -inf score to padding nodes
        arr = torch.arange(0, graph_sizes.max(), dtype=torch.long, device=scores.device)
        pad_node_mask = arr[None] >= n_nodes_per_graph[:, None]
        assert pad_node_mask.size() == (n_pointers, max_nodes)
        scores.masked_fill_(pad_node_mask, -float("inf"))

        return scores
