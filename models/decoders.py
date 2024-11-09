"""Graph decoders."""
import manifolds
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
from layers.att_layers import GraphAttentionLayer
from layers.layers import GraphConvolution, Linear


class Decoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """
    #这是所有解码器的基类，它继承了nn.Module。c是一个参数，用于超球面中的曲率。
    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c
    #这是基类中的decode方法，它可以基于输入x和邻接矩阵adj计算分类的概率。
    def decode(self, x, adj):
        if self.decode_adj:
            input = (x, adj)
            probs, _ = self.cls.forward(input)
        else:
            probs = self.cls.forward(x)
        return probs

#这个类实现了图卷积解码器。它初始化了一个GraphConvolution层来执行图卷积操作，并设置decode_adj为True，表示该解码器需要邻接矩阵。
class GCNDecoder(Decoder):
    """
    Graph Convolution Decoder.
    """

    def __init__(self, c, args):
        super(GCNDecoder, self).__init__(c)
        #act = lambda x: x
        self.cls = GraphConvolution(args.dim, args.n_classes, args.dropout, lambda x: x, args.bias)
        self.decode_adj = True

#这个类实现了图注意力解码器，它初始化了一个GraphAttentionLayer层来执行图注意力操作。
class GATDecoder(Decoder):
    """
    Graph Attention Decoder.
    """

    def __init__(self, c, args):
        super(GATDecoder, self).__init__(c)
        self.cls = GraphAttentionLayer(args.dim, args.n_classes, args.dropout, F.elu, args.alpha, 1, True)
        self.decode_adj = True



#这个类实现了线性解码器，它适用于超球面或欧几里得节点分类模型，可以处理多种流形。它初始化了一个Linear层来执行线性变换。
class LinearDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, c, args):
        super(LinearDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim
        self.output_dim = args.n_classes
        self.bias = args.bias
        self.cls = Linear(self.input_dim, self.output_dim, args.dropout, lambda x: x, self.bias)
        self.decode_adj = False
#这个方法首先将输入x从超球面映射到切平面，然后调用基类的decode方法来计算分类概率。
    def decode(self, x, adj):
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        return super(LinearDecoder, self).decode(h, adj)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
                self.input_dim, self.output_dim, self.bias, self.c
        )




class HyperbolicHGNNDecoder(Decoder):
    def __init__(self, c, args):
        super(HyperbolicHGNNDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim
        self.output_dim = args.n_classes
        self.bias = args.bias
        self.cls = HGNN_conv_1(self.input_dim, self.output_dim, self.bias)
        self.decode_adj = True

    def decode(self, x, adj):
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        return self.cls(h, adj)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
                self.input_dim, self.output_dim, self.bias, self.c
        )

class HGNN_conv_1(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv_1, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x,adj):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = adj.matmul(x)
        return x


class HGNN_fc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HGNN_fc, self).__init__()
        self.fc = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)


class HGNN_embedding(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.5):
        super(HGNN_embedding, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv_1(in_ch, n_hid)
        self.hgc2 = HGNN_conv_1(n_hid, n_hid)

    def forward(self, x, adj):
        x = F.relu(self.hgc1(x, adj))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.hgc2(x, adj))
        return x


class HGNN_classifier(nn.Module):
    def __init__(self, n_hid, n_class):
        super(HGNN_classifier, self).__init__()
        self.fc1 = nn.Linear(n_hid, n_class)

    def forward(self, x):
        x = self.fc1(x)
        return x


#这是一个字典，映射模型名称到相应的解码器类，用于在主程序中创建解码器实例。
model2decoder = {
    'GCN': GCNDecoder,
    'GAT': GATDecoder,
    'HNN': LinearDecoder,
    'HGCN': LinearDecoder,
    'MLP': LinearDecoder,
    'Shallow': LinearDecoder,
    #"HGNN":GATDecoder
    "HGNN":HyperbolicHGNNDecoder,
    "DHHNN":HyperbolicHGNNDecoder,

}

