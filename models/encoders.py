"""Graph encoders."""

import numpy as np
import torch
import torch.nn as nn

import manifolds
from layers.att_layers import GraphAttentionLayer
import layers.hyp_layers as hyp_layers
from layers.layers import GraphConvolution, Linear, get_dim_act

'''
#这里定义了一个编码器基类，它继承了nn.Module。self.c是一个参数，可能是超球面的曲率。
class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c
#这里定义了基类的encode方法，它可以根据self.encode_graph的值来决定是否需要图结构信息作为输入。
    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output
'''
class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)  # Unpack all values
        else:
            output = self.layers.forward(x)
        return output
#定义一个继承自Encoder的MLP类，表示多层感知机。它有一个层列表layers。
class MLP(Encoder):
    """
    Multi-layer perceptron.
    """

    def __init__(self, c, args):
        super(MLP, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        layers = []
        #遍历每一层，添加线性层到layers列表中。
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        #将层列表转化为一个nn.Sequential模块，并设置self.encode_graph为False
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False

#定义了一个继承自Encoder的HNN类，表示超球面神经网络。它也有一个层列表hnn_layers。
class HNN(Encoder):
    """
    Hyperbolic Neural Networks.
    """

    def __init__(self, c, args):
        super(HNN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, _ = hyp_layers.get_dim_act_curv(args)
        hnn_layers = []
        #遍历每一层，添加超球面神经网络层到hnn_layers列表中。
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hnn_layers.append(
                    hyp_layers.HNNLayer(
                            self.manifold, in_dim, out_dim, self.c, args.dropout, act, args.bias)
            )
        #将层列表转化为一个nn.Sequential模块，并设置self.encode_graph为False。
        self.layers = nn.Sequential(*hnn_layers)
        self.encode_graph = False
    #这里重新定义了encode方法，该方法将输入从欧几里得空间映射到超球面空间，并在超球面空间中进行编码。
    def encode(self, x, adj):
        x_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        return super(HNN, self).encode(x_hyp, adj)
#定义了一个继承自Encoder的GCN类，表示图卷积网络。它也有一个层列表gc_layers。
class GCN(Encoder):
    """
    Graph Convolution Networks.
    """

    def __init__(self, c, args):
        super(GCN, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gc_layers = []
        #遍历每一层，添加图卷积层到gc_layers列表中。
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gc_layers.append(GraphConvolution(in_dim, out_dim, args.dropout, act, args.bias))
        #将层列表转化为一个nn.Sequential模块，并设置self.encode_graph为True。
        self.layers = nn.Sequential(*gc_layers)
        self.encode_graph = True

#HGCN类继承自Encoder基类。这里，c可能表示超球面的曲率，args包含模型的参数和配置。
class HGCN(Encoder):
    """
    Hyperbolic-GCN.
    """
    def __init__(self, c, args):
        super(HGCN, self).__init__(c)
        #通过args.manifold，从manifolds模块中获取对应的超球面。超球面用于定义超球面上的数学运算，如加法、指数映射、对数映射等。
        self.manifold = getattr(manifolds, args.manifold)()
        #args.num_layers > 1确保网络至少有一个隐藏层。get_dim_act_curv函数返回每一层的维度、激活函数和曲率。
        assert args.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        #self.curvatures是一个包含每一层的曲率的列表。最后一层的曲率是self.c。
        self.curvatures.append(self.c)
        #对于每一层，都创建了一个HyperbolicGraphConvolution层，并将它添加到hgc_layers列表中。这个层执行超球面上的图卷积操作。
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    hyp_layers.HyperbolicGraphConvolution(
                            self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att, args.local_agg
                    )
            )
        #将hgc_layers列表转换为nn.Sequential模块。设置self.encode_graph为True，表示这个模型需要图结构作为输入。
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True
    #在encode方法中，输入x首先被投影到切平面上，然后被映射到超球面上，最后在超球面上进行编码。这个方法返回编码后的结果。
    def encode(self, x, adj):
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        return super(HGCN, self).encode(x_hyp, adj)


class GAT(Encoder):
    """
    Graph Attention Networks.
    """

    def __init__(self, c, args):
        super(GAT, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gat_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            assert dims[i + 1] % args.n_heads == 0
            out_dim = dims[i + 1] // args.n_heads
            concat = True
            gat_layers.append(
                    GraphAttentionLayer(in_dim, out_dim, args.dropout, act, args.alpha, args.n_heads, concat))
        self.layers = nn.Sequential(*gat_layers)
        self.encode_graph = True


class Shallow(Encoder):
    """
    Shallow Embedding method.
    Learns embeddings or loads pretrained embeddings and uses an MLP for classification.
    """

    def __init__(self, c, args):
        super(Shallow, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.use_feats = args.use_feats
        weights = torch.Tensor(args.n_nodes, args.dim)
        if not args.pretrained_embeddings:
            weights = self.manifold.init_weights(weights, self.c)
            trainable = True
        else:
            weights = torch.Tensor(np.load(args.pretrained_embeddings))
            assert weights.shape[0] == args.n_nodes, "The embeddings you passed seem to be for another dataset."
            trainable = False
        self.lt = manifolds.ManifoldParameter(weights, trainable, self.manifold, self.c)
        self.all_nodes = torch.LongTensor(list(range(args.n_nodes)))
        layers = []
        if args.pretrained_embeddings is not None and args.num_layers > 0:
            # MLP layers after pre-trained embeddings
            dims, acts = get_dim_act(args)
            if self.use_feats:
                dims[0] = args.feat_dim + weights.shape[1]
            else:
                dims[0] = weights.shape[1]
            for i in range(len(dims) - 1):
                in_dim, out_dim = dims[i], dims[i + 1]
                act = acts[i]
                layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False

    def encode(self, x, adj):
        h = self.lt[self.all_nodes, :]
        if self.use_feats:
            h = torch.cat((h, x), 1)
        return super(Shallow, self).encode(h, adj)

'''
class HGNN(Encoder):
    """
    Hyperbolic Graph Neural Networks using Hypergraph Convolution.
    """

    def __init__(self, c, args):
        super(HGNN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()

        assert args.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)

        hg_layers = []

        # Initial HNN Layers for embedding into hyperbolic space
        for i in range(len(dims) - 2):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i] if i < len(acts) else lambda x: x  # Fallback to identity function if acts is too short
            hg_layers.append(
                hyp_layers.HyperbolicHGNNConv(
                    self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att,args.local_agg
                )
            )

        # Hypergraph Convolution in Hyperbolic space
        c_in, c_out = self.curvatures[-2], self.curvatures[-1]
        in_dim, out_dim = dims[-2], dims[-1]
        act = acts[-2] if len(acts) > 1 else lambda x: x  # Fallback to identity function if acts is too short
        hg_layers.append(
            hyp_layers.HyperbolicHGNNConv(
                self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att, args.local_agg
            )
        )

        self.layers = nn.Sequential(*hg_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        return super(HGNN, self).encode(x_hyp, adj)
'''

class DHHNN(Encoder):
    def __init__(self, c, args):
        super(DHHNN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        hgc_layers = []
        self.args = args
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    hyp_layers.HyperDHHNNConvolution(
                            self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att, args.local_agg,self.args
                    )

            )

        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        return super(DHHNN, self).encode(x_hyp, adj)


