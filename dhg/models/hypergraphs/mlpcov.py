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

    def forward(self, x: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        x = torch.nn.functional.relu(self.hidden(x))
        x = self.predict(x)
        return x
