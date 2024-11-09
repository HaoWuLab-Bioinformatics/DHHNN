"""Hyperbolic layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module

from hypergraph import Hypergraph
from model.attention.CBAM import CBAMBlock

import torch


from layers.att_layers import DenseAtt
def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
        n_curvatures = args.num_layers
    else:
        n_curvatures = args.num_layers - 1
    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.hyp_act.forward(h)
        output = h, adj
        return output

class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        if self.use_att:
            if self.local_agg:
                x_local_tangent = []
                for i in range(x.size(0)):
                    x_local_tangent.append(self.manifold.logmap(x[i], x, c=self.c))
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj)
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                output = self.manifold.proj(self.manifold.expmap(x, support_t, c=self.c), c=self.c)
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = torch.matmul(adj_att, x_tangent)
        else:
            support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        x_logmap0 = self.manifold.logmap0(x, c=self.c_in)
        xt = self.act(x_logmap0)  # 确保 act 不是 in-place 操作
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        xt = self.manifold.expmap0(xt, c=self.c_out)
        return self.manifold.proj(xt, c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )


import math
from turtle import forward
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv, GATConv


class HConstructor(nn.Module):
    def __init__(self, num_edges, f_dim, iters=1, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_edges = num_edges
        self.edges = None
        self.iters = iters
        self.eps = eps
        self.scale = f_dim ** -0.5
        # self.scale = 1

        self.edges_mu = nn.Parameter(torch.randn(1, f_dim))
        self.edges_logsigma = nn.Parameter(torch.zeros(1, f_dim))
        init.xavier_uniform_(self.edges_logsigma)

        self.to_q = nn.Linear(f_dim, f_dim)
        self.to_k = nn.Linear(f_dim, f_dim)
        self.to_v = nn.Linear(f_dim, f_dim)

        self.gru = nn.GRUCell(f_dim, f_dim)

        hidden_dim = max(f_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(f_dim + f_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, f_dim)
        )

        self.norm_input = nn.LayerNorm(f_dim)
        self.norm_edgs = nn.LayerNorm(f_dim)
        self.norm_pre_ff = nn.LayerNorm(f_dim)

    def mask_attn(self, attn, k):
        indices = torch.topk(attn, k).indices
        mask = torch.zeros(attn.shape).bool().to(attn.device)
        for i in range(attn.shape[0]):
            mask[i][indices[i]] = 1
        return attn.mul(mask)

    def ajust_edges(self, s_level, args):
        if args.stage != 'train':
            return

        if s_level > args.up_bound:
            self.num_edges = self.num_edges + 1
        elif s_level < args.low_bound:
            self.num_edges = self.num_edges - 1
            self.num_edges = max(self.num_edges, args.min_num_edges)
        else:
            return

    def forward(self, inputs, args):
        n, d, device = *inputs.shape, inputs.device
        n_s = self.num_edges

        if True:
            # if self.edges is None:
            mu = self.edges_mu.expand(n_s, -1)
            sigma = self.edges_logsigma.exp().expand(n_s, -1)
            edges = mu + sigma * torch.randn(mu.shape, device=device)
        else:
            edges = self.edges

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        k = F.relu(k)
        v = F.relu(v)

        for _ in range(self.iters):
            edges = self.norm_edgs(edges)

            # 求结点相对于边的softmax
            q = self.to_q(edges)
            q = F.relu(q)

            dots = torch.einsum('ni,ij->nj', q, k.T) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=1, keepdim=True)
            attn = self.mask_attn(attn, args.k_n)  # 这个决定边的特征从哪些结点取

            # 更新超边特征
            updates = torch.einsum('in,nf->if', attn, v)
            edges = torch.cat((edges, updates), dim=1)
            edges = self.mlp(edges)

            # 按边相对于结点的softmax（更新边之后）
            q = self.to_q(inputs)
            k = self.to_k(edges)
            k = F.relu(k)
            v = F.relu(v)

            dots = torch.einsum('ni,ij->nj', q, k.T) * self.scale
            attn_v = dots.softmax(dim=1)
            attn_v = self.mask_attn(attn_v, args.k_e)  # 这个决定一个结点属于多少条边
            H = attn_v

            # 计算边的饱和度
            cc = H.ceil().abs()
            de = cc.sum(dim=0)
            empty = (de == 0).sum()
            s_level = 1 - empty / n_s

            self.ajust_edges(s_level, args)

            print("Num edges is: {}; Satuation level is: {}".format(self.num_edges, s_level))

        self.edges = edges

        return edges, H, dots


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, num_edges, bias=True):
        super(HGNN_conv, self).__init__()

        self.HConstructor = HConstructor(num_edges, in_ft)

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(in_ft, out_ft))
        self.mlp.append(nn.Linear(out_ft, out_ft))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, args):
        edges, H, H_raw = self.HConstructor(x, args)
        edges = edges.matmul(self.weight)
        if self.bias is not None:
            edges = edges + self.bias
        nodes = H.matmul(edges)
        # x = self.mlp[0](x) + self.mlp[1](nodes)
        x = x + nodes
        return x, H, H_raw


class HGNN_classifier(nn.Module):
    def __init__(self, args, dropout=0.5):
        super(HGNN_classifier, self).__init__()
        in_dim = args.in_dim
        hid_dim = args.hid_dim
        out_dim = args.out_dim
        num_edges = args.num_edges
        self.conv_number = args.conv_number

        self.dropout = dropout

        # self.linear_backbone = nn.Linear(in_dim,hid_dim)

        self.linear_backbone = nn.ModuleList()
        self.linear_backbone.append(nn.Linear(in_dim, hid_dim))
        self.linear_backbone.append(nn.Linear(hid_dim, hid_dim))
        self.linear_backbone.append(nn.Linear(hid_dim, hid_dim))

        self.gcn_backbone = nn.ModuleList()
        self.gcn_backbone.append(GCNConv(in_dim, hid_dim))
        self.gcn_backbone.append(GCNConv(hid_dim, hid_dim))

        self.convs = nn.ModuleList()
        self.transfers = nn.ModuleList()

        for i in range(self.conv_number):
            self.convs.append(HGNN_conv(hid_dim, hid_dim, num_edges))
            self.transfers.append(nn.Linear(hid_dim, hid_dim))

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.conv_number * hid_dim, out_dim),
        )

    def forward(self, data, args):

        if args.backbone == 'linear':
            #x = data['fts']

            # x = self.linear_backbone[0](x)
            data = F.relu(self.linear_backbone[0](data))
            data = F.relu(self.linear_backbone[1](data))
            data = self.linear_backbone[2](data)
        elif args.backbone == 'gcn':
            x = data['fts']
            edge_index = data['edge_index']
            x = self.gcn_backbone[0](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.gcn_backbone[1](x, edge_index)

        tmp = []
        H = []
        H_raw = []
        for i in range(self.conv_number):
            data, h, h_raw = self.convs[i](data, args)
            data = F.relu(data)
            data = F.dropout(data, training=self.training)
            if args.transfer == 1:
                data = self.transfers[i](data)
                data = F.relu(data)
            tmp.append(data)
            H.append(h)
            H_raw.append(h_raw)

        data = torch.cat(tmp, dim=1)

        out = self.classifier(data)
        return out, data, H, H_raw


class GCN(nn.Module):
    def __init__(self, args, layer_number=2):

        in_dim = args.in_dim
        hid_dim = args.hid_dim
        out_dim = args.out_dim

        super(GCN, self).__init__()
        # graph convolution
        self.convs = nn.ModuleList()

        self.convs.append(GCNConv(in_dim, hid_dim))
        for i in range(1, layer_number):
            self.convs.append(GCNConv(hid_dim, hid_dim))

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, data, args):
        x = data['fts']
        edge_index = data['edge_index']

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

        out = self.classifier(x)
        return out, x, None, None


class GAT(nn.Module):
    def __init__(self, args, layer_number=2):
        super(GAT, self).__init__()

        in_dim = args.in_dim
        hid_dim = args.hid_dim
        out_dim = args.out_dim

        # graph convolution
        self.convs = nn.ModuleList()

        self.convs.append(GATConv(in_dim, hid_dim))
        for i in range(1, layer_number):
            self.convs.append(GATConv(hid_dim, hid_dim))

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, data, args):
        x = data['fts']
        edge_index = data['edge_index']

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

        out = self.classifier(x)
        return out, x, None, None

class DHHNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, num_edges, bias=True):
        super(DHHNN_conv, self).__init__()

        self.HConstructor = HConstructor(num_edges, in_ft)

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(in_ft, out_ft))
        self.mlp.append(nn.Linear(out_ft, out_ft))
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, args):
        edges, H, H_raw = self.HConstructor(x, args)
        edges = edges.matmul(self.weight)
        if self.bias is not None:
            edges = edges + self.bias
        nodes = H.matmul(edges)
        # x = self.mlp[0](x) + self.mlp[1](nodes)
        x = x + nodes
        return x, H, H_raw


from torch.nn import Module
import torch
import math
import torch.nn.functional as F

class MultiHeadAttention(Module):
    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 device: str,
                 mask: bool = False,
                 dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()

        self.W_q = torch.nn.Linear(d_model, q * h)
        self.W_k = torch.nn.Linear(d_model, q * h)
        self.W_v = torch.nn.Linear(d_model, v * h)

        self.W_o = torch.nn.Linear(v * h, d_model)

        self.device = device
        self._h = h
        self._q = q

        self.mask = mask
        self.dropout = torch.nn.Dropout(p=dropout)
        self.score = None

    def forward(self, x, stage):
        Q = torch.cat(self.W_q(x).chunk(self._h, dim=-1), dim=0)
        K = torch.cat(self.W_k(x).chunk(self._h, dim=-1), dim=0)
        V = torch.cat(self.W_v(x).chunk(self._h, dim=-1), dim=0)

        score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self._q)
        self.score = score

        if self.mask and stage == 'train':
            mask = torch.ones_like(score[0])
            mask = torch.tril(mask, diagonal=0)
            score = torch.where(mask > 0, score, torch.Tensor([-2 ** 32 + 1]).expand_as(score[0]).to(self.device))

        score = F.softmax(score, dim=-1)

        attention = torch.matmul(score, V)

        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        self_attention = self.W_o(attention_heads)

        return self_attention, self.score

from model.attention.MobileViTv2Attention import MobileViTv2Attention
class HyperDHHNNConvolution(nn.Module):

    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg,args):
        super(HyperDHHNNConvolution, self).__init__()
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择设备 CPU or GPU
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        #self.eca_attention = ECAAttention(kernel_size=6)
        #self.hyp_hid = HypLinear(manifold, in_features, 512, c_in, dropout=dropout, use_bias=use_bias)
        #self.hyp_linear = HypLinear(manifold, 512, out_features, c_in, dropout=dropout, use_bias=use_bias)
        #self.sa=MobileViTv2Attention(d_model=128)
        self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)
        self.args = args
        self.classifier = HGNN_classifier(args)
        self.dropout = dropout

    def forward(self, input):
        x, adj = input

        #h = self.hyp_hid.forward(x)
        #h=self.hyp_linear.forward(h)
        h = self.linear.forward(x)
        #h=self.sa(h,stage='train')
        # 注意，这里我们使用相同的输入作为查询、键和值
        attention_output = F.scaled_dot_product_attention(h, h, h)
        # 将注意力输出与输入相加，类似于残差连接
        h = h + attention_output
        #h=self.eca_attention(h)
        h = self.agg.forward(h, adj)
        #print(h.shape)
        h = self.hyp_act.forward(h)
        out, abc, H, H_raw = self.classifier.forward(h,self.args)
        #output = h, adj
        #return output
        output=h, adj, out, abc, H, H_raw
        return h, adj

        '''
        h = self.hyp_hid(x)

        h = self.agg.forward(h, adj)
        h = self.hyp_act.forward(h)
        h = F.dropout(h, self.dropout)
        # 注意，这里我们使用相同的输入作为查询、键和值
        attention_output = F.scaled_dot_product_attention(h, h, h)
        # 将注意力输出与输入相加，类似于残差连接
        h = h + attention_output
        #h = self.hyp_hid(h)
        # Hyperbolic linear transformation
        h = self.hyp_linear(h)
        #h = self.hyp_hid(x,adj)
        # Hyperbolic aggregation
        h = self.agg.forward(h, adj)
        # Hyperbolic activation
        h = self.hyp_act.forward(h)
        out, abc, H, H_raw = self.classifier.forward(h,self.args)
        #output = h, adj
        #return output
        output=h, adj, out, abc, H, H_raw
        
        return output
        '''
class HyperbolicHGNNConv(nn.Module):
    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg,use_bn: bool = True,):
        super(HyperbolicHGNNConv, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.use_att = use_att
        self.local_agg = local_agg
        self.dropout = dropout
        #self.bn = nn.BatchNorm1d(128) if use_bn else None
        #self.cn = nn.BatchNorm1d(out_features) if use_bn else None
        # Hyperbolic linear layer
        self.hyp_hid = HypLinear(manifold, in_features, 512, c_in, dropout=dropout, use_bias=use_bias)
        self.hyp_linear = HypLinear(manifold, 512, out_features, c_in, dropout=dropout, use_bias=use_bias)

        # Hyperbolic aggregation layer
        self.hyp_agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg)

        # Hyperbolic activation function
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.hyp_hid(x)

        h = self.hyp_agg(h, adj)
        h = self.hyp_act(h)
        h = F.dropout(h, self.dropout)
        #h = self.hyp_hid(h)
        # Hyperbolic linear transformation
        h = self.hyp_linear(h)
        #h = self.hyp_hid(x,adj)
        # Hyperbolic aggregation
        h = self.hyp_agg(h, adj)
        # Hyperbolic activation
        h = self.hyp_act(h)
        return h, adj
