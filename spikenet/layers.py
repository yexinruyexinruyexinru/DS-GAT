import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, concat=False,bias=False):
        super(GAT, self).__init__()
        self.in_features = in_dim
        self.out_features = out_dim
        self.concat = concat
        self.aggr1 = nn.Linear(5, 1, bias=False)
        self.aggr2 = nn.Linear(2, 1, bias=False)
        self.lin_l = nn.Linear(in_dim, out_dim, bias=bias)
        self.lin_r = nn.Linear(in_dim, out_dim, bias=bias)

    def aggragetor(self,one_neigh_x):
        neigh_x_t=one_neigh_x.transpose(0,1)
        # out=torch.empty(1)
        if len(one_neigh_x)==5:
            out=self.aggr1(neigh_x_t)
        elif len(one_neigh_x)==2:
            out=self.aggr2(neigh_x_t)
        return out.transpose(0,1)

    def forward(self, x, neigh_x):
        x_lis=[]
        neigh_x_lis=[]
        for i,t_x in enumerate(x):
            t_neigh_x=neigh_x[i]
            for j,h in enumerate(t_neigh_x):
                x_lis.append(t_x[j])
                aggr = self.aggragetor(h)
                neigh_x_lis.append(aggr.view(-1))
        x=torch.stack(x_lis, dim=0)
        neigh_x = torch.stack(neigh_x_lis, dim=0)
        x = self.lin_l(x)
        neigh_x = self.lin_r(neigh_x)
        out = torch.cat([x, neigh_x], dim=1) if self.concat else x + neigh_x
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class SAGEAggregator(nn.Module):
    def __init__(self, in_features, out_features,
                 aggr='mean',
                 concat=False,
                 bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.aggr = aggr
        self.aggregator = {'mean': torch.mean, 'sum': torch.sum}[aggr]
        self.lin_l = nn.Linear(in_features, out_features, bias=bias)
        self.lin_r = nn.Linear(in_features, out_features, bias=bias)

    def aggragetor1(self,one_neigh_x):
        neigh_x_t=one_neigh_x.transpose(0,1)
        if len(one_neigh_x)==3:
            out=self.aggr1(neigh_x_t)
        elif len(one_neigh_x)==2:
            out=self.aggr2(neigh_x_t)
        return out.transpose(0,1)

    def forward(self, x, neigh_x):
        if not isinstance(x, torch.Tensor):
            x = torch.cat(x, dim=0)
        if not isinstance(neigh_x, torch.Tensor):
            neigh_x = torch.cat([self.aggregator(h, dim=1)
                                for h in neigh_x], dim=0)
        else:
            neigh_x = self.aggregator(neigh_x, dim=1)
        x = self.lin_l(x)
        neigh_x = self.lin_r(neigh_x)
        out = torch.cat([x, neigh_x], dim=1) if self.concat else x + neigh_x
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, aggr={self.aggr})"
