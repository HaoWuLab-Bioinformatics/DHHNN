import torch
import torch.nn as nn

from dhg.structure.hypergraphs import Hypergraph



class mlpNet(nn.Module):
    def __init__(self,
                 n_feature:int,
                 n_hidden:int,
                 n_output:int,
                 ):
        super().__init__()
        # 两层感知机
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        X = self.theta(X)
        if self.bn is not None:
            X = self.bn(X)
        X = hg.v2v(X, aggr="mean")
        if not self.is_last:
            X = self.drop(self.act(X))
        return X
