# --------------------------------------------------------
# Utility functions for Hypergraph
#
# Author: Yifan Feng
# Date: November 2018
# --------------------------------------------------------
import numpy as np


import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
# from utils import load_data, load_graph
# from GNN import GNNLayer
# from evaluation import eva
# from collections import Counter
#from prognn import ProGNN
# from deeprobust.graph.data import Dataset, PrePtbDataset
# from deeprobust.graph.utils import preprocess, encode_onehot, get_train_val_test
# from deeprobust.graph.defense import GCN
from sklearn.model_selection import train_test_split
#torch.cuda.set_device(1)
import scipy.sparse as sp

import time
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from deeprobust.graph.utils import accuracy
# from deeprobust.graph.defense.pgd import PGD, prox_operators
import warnings
#import opt
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from sklearn.metrics import pairwise_distances
class ProGNN:
    """ ProGNN (Properties Graph Neural Network). See more details in Graph Structure Learning for Robust Graph Neural Networks, KDD 2020, https://arxiv.org/abs/2005.10203.

    Parameters
    ----------
    model:
        model: The backbone GNN model in ProGNN
    args:
        model configs
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
    See details in https://github.com/ChandlerBang/Pro-GNN.

    """

    def __init__(self, model, args, device):
        self.device = device
        self.args = args
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_graph = None
        self.weights = None
        self.estimator = None
        self.model = model.to(device)

    def fit(self, features, adj, labels, idx_train, idx_val, **kwargs):
        """Train Pro-GNN.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices
        """
        args = self.args

        self.optimizer = optim.Adam(self.model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
        estimator = EstimateAdj(adj, symmetric=args.symmetric, device=self.device).to(self.device)
        self.estimator = estimator
        self.optimizer_adj = optim.SGD(estimator.parameters(),
                              momentum=0.9, lr=args.lr_adj)

        self.optimizer_l1 = PGD(estimator.parameters(),
                        proxs=[prox_operators.prox_l1],
                        lr=args.lr_adj, alphas=[args.alpha])

        # warnings.warn("If you find the nuclear proximal operator runs too slow on Pubmed, you can  uncomment line 67-71 and use prox_nuclear_cuda to perform the proximal on gpu.")
        # if args.dataset == "pubmed":
        #     self.optimizer_nuclear = PGD(estimator.parameters(),
        #               proxs=[prox_operators.prox_nuclear_cuda],
        #               lr=args.lr_adj, alphas=[args.beta])
        # else:
        warnings.warn("If you find the nuclear proximal operator runs too slow, you can modify line 77 to use prox_operators.prox_nuclear_cuda instead of prox_operators.prox_nuclear to perform the proximal on GPU. See details in https://github.com/ChandlerBang/Pro-GNN/issues/1")
        self.optimizer_nuclear = PGD(estimator.parameters(),
                  proxs=[prox_operators.prox_nuclear_cuda],
                  lr=args.lr_adj, alphas=[args.beta])

        # Train model
        t_total = time.time()
        for epoch in range(args.epochs):
            if args.only_gcn:
                self.train_gcn(epoch, features, estimator.estimated_adj,
                        labels, idx_train, idx_val)
            else:
                for i in range(int(args.outer_steps)):
                    self.train_adj(epoch, features, adj, labels,
                            idx_train, idx_val)

                for i in range(int(args.inner_steps)):
                    self.train_gcn(epoch, features, estimator.estimated_adj,
                            labels, idx_train, idx_val)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print(args)

        # Testing
        print("picking the best model according to validation performance")
        self.model.load_state_dict(self.weights)
        return self.best_graph

    def train_gcn(self, epoch, features, adj, labels, idx_train, idx_val):
        args = self.args
        estimator = self.estimator
        adj = estimator.normalize()

        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        self.optimizer.step()

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        output = self.model(features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print('\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())

        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'acc_train: {:.4f}'.format(acc_train.item()),
                      'loss_val: {:.4f}'.format(loss_val.item()),
                      'acc_val: {:.4f}'.format(acc_val.item()),
                      'time: {:.4f}s'.format(time.time() - t))



    def train_adj(self, epoch, features, adj, labels, idx_train, idx_val):
        estimator = self.estimator
        args = self.args
        if args.debug:
            print("\n=== train_adj ===")
        t = time.time()
        estimator.train()
        self.optimizer_adj.zero_grad()

        loss_l1 = torch.norm(estimator.estimated_adj, 1)
        loss_fro = torch.norm(estimator.estimated_adj - adj, p='fro')
        normalized_adj = estimator.normalize()

        if args.lambda_:
            loss_smooth_feat = self.feature_smoothing(estimator.estimated_adj, features)
        else:
            loss_smooth_feat = 0 * loss_l1

        output = self.model(features, normalized_adj)

        loss_gcn = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

        loss_symmetric = torch.norm(estimator.estimated_adj \
                        - estimator.estimated_adj.t(), p="fro")

        loss_diffiential =  loss_fro + args.gamma * loss_gcn + args.lambda_ * loss_smooth_feat + args.phi * loss_symmetric

        loss_diffiential.backward()

        self.optimizer_adj.step()
        loss_nuclear =  0 * loss_fro
        if args.beta != 0:
            self.optimizer_nuclear.zero_grad()
            self.optimizer_nuclear.step()
            loss_nuclear = prox_operators.nuclear_norm

        self.optimizer_l1.zero_grad()
        self.optimizer_l1.step()

        total_loss = loss_fro \
                    + args.gamma * loss_gcn \
                    + args.alpha * loss_l1 \
                    + args.beta * loss_nuclear \
                    + args.phi * loss_symmetric

        estimator.estimated_adj.data.copy_(torch.clamp(
                  estimator.estimated_adj.data, min=0, max=1))

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        normalized_adj = estimator.normalize()
        output = self.model(features, normalized_adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch+1),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = normalized_adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = normalized_adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())

        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_fro: {:.4f}'.format(loss_fro.item()),
                      'loss_gcn: {:.4f}'.format(loss_gcn.item()),
                      'loss_feat: {:.4f}'.format(loss_smooth_feat.item()),
                      'loss_symmetric: {:.4f}'.format(loss_symmetric.item()),
                      'delta_l1_norm: {:.4f}'.format(torch.norm(estimator.estimated_adj-adj, 1).item()),
                      'loss_l1: {:.4f}'.format(loss_l1.item()),
                      'loss_total: {:.4f}'.format(total_loss.item()),
                      'loss_nuclear: {:.4f}'.format(loss_nuclear.item()))


    def test(self, features, labels, idx_test):
        """Evaluate the performance of ProGNN on test set
        """
        print("\t=== testing ===")
        self.model.eval()
        adj = self.best_graph
        if self.best_graph is None:
            adj = self.estimator.normalize()
        output = self.model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("\tTest set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    def feature_smoothing(self, adj, X):
        adj = (adj.t() + adj)/2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj

        r_inv = r_inv  + 1e-3
        r_inv = r_inv.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        # L = r_mat_inv @ L
        L = r_mat_inv @ L @ r_mat_inv

        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)
        return loss_smooth_feat


class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, adj, symmetric=False, device='cpu'):
        super(EstimateAdj, self).__init__()
        n = adj.shape[0]
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n))
        self._init_estimation(adj)
        self.symmetric = symmetric
        self.device = device

    def _init_estimation(self, adj):
        with torch.no_grad():
            n = adj.shape[0]
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

    def normalize(self):

        if self.symmetric:
            adj = (self.estimated_adj + self.estimated_adj.t())
        else:
            adj = self.estimated_adj

        normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx




    
def robust(x,y,adj):
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='dblp')
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug mode')
    parser.add_argument('--only_gcn', action='store_true',
                        default=False, help='test the performance of gcn without other components')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=15, help='Random seed.')

    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--ptb_rate', type=float, default=0.05, help="noise ptb_rate")
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--alpha', type=float, default=5e-4, help='weight of l1 norm')
    parser.add_argument('--beta', type=float, default=1.5, help='weight of nuclear norm')
    parser.add_argument('--gamma', type=float, default=1, help='weight of l2 norm')
    parser.add_argument('--lambda_', type=float, default=0, help='weight of feature smoothing')
    parser.add_argument('--phi', type=float, default=0, help='weight of symmetric loss')
    parser.add_argument('--inner_steps', type=int, default=2, help='steps for inner optimization')
    parser.add_argument('--outer_steps', type=int, default=1, help='steps for outer optimization')
    parser.add_argument('--lr_adj', type=float, default=0.01, help='lr for training adj')
    parser.add_argument('--symmetric', action='store_true', default=False,
                        help='whether use symmetric matrix')

    args = parser.parse_args(args=[])
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")




    features = sp.coo_matrix(x)
    labels = y
    adj = sp.coo_matrix(adj)
    idx_train, idx_val, idx_test = get_train_val_test(nnodes=adj.shape[0], val_size=0.1, test_size=0.1,
                                                      stratify=encode_onehot(labels), seed=15)

    # robust模型

    model = GCN(nfeat=features.shape[1],
                 nhid=args.hidden,
                 nclass=labels.max().item() + 1,
                 dropout=args.dropout, device=device)
    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, device=device)
    prognn = ProGNN(model, args, device)
    adj = prognn.fit(features, adj, labels, idx_train, idx_val)
    prognn.test(features, labels, idx_test)
    return adj


#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
'''
def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)

    return dist_mat
'''

'''
def Eu_dis(x):
    """
    Calculate the distance among each row of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    # 如果x是稀疏的，将其转换为密集数组
    if sp.issparse(x):
        x = x.toarray()

    aa = np.sum(x * x, axis=1).reshape(-1, 1)
    ab = np.dot(x, x.T)
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)

    return dist_mat
'''
def Eu_dis(x):
    """
    Calculate the distance among each row of x using efficient vectorized operations.
    """
    if sp.issparse(x):
        x = x.toarray()
    return pairwise_distances(x, metric='euclidean')





def feature_concat(*F_list, normal_col=False):
    """
    Concatenate multiple modality feature. If the dimension of a feature matrix is more than two,
    the function will reduce it into two dimension(using the last dimension as the feature dimension,
    the other dimension will be fused as the object dimension)
    :param F_list: Feature matrix list
    :param normal_col: normalize each column of the feature
    :return: Fused feature matrix
    """
    features = None
    for f in F_list:
        if f is not None and f != []:
            # deal with the dimension that more than two
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            # normal each column
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """

    H = None
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):
    '''
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    
    #H = H.cpu().numpy()
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        #G = DV2 * H * W * invDE * HT * DV2
        G = DV2 * H * HT * DV2
        return G
    '''

    '''
    H = sp.csr_matrix(H)
    n_edge = H.shape[1]
    W = np.ones(n_edge)
    DV = np.array(H.sum(axis=1)).flatten()
    DE = np.array(H.sum(axis=0)).flatten()

    invDE = sp.diags(1.0 / DE)
    DV2 = sp.diags(np.sqrt(1.0 / DV))
    W = sp.diags(W)
    HT = H.T

    if variable_weight:
        DV2_H = DV2.dot(H)
        invDE_HT_DV2 = invDE.dot(HT).dot(DV2)
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2.dot(H).dot(HT).dot(DV2)
        return G
        '''
    H = sp.csr_matrix(H)
    n_edge = H.shape[1]
    W = np.ones(n_edge)
    DV = np.array(H.sum(axis=1)).flatten()
    DE = np.array(H.sum(axis=0)).flatten()

    invDE = sp.diags(1.0 / DE)
    DV2 = sp.diags(np.sqrt(1.0 / DV))

    if variable_weight:
        DV2_H = DV2.dot(H)
        invDE_HT_DV2 = invDE.dot(H.T).dot(DV2)
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2.dot(H).dot(H.T).dot(DV2)
        return G

def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    '''
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[node_idx] ** 2 / (m_prob * avg_dis) ** 2)

                #H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H
    '''

    '''
    n_obj = dis_mat.shape[0]
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    avg_dis = np.mean(dis_mat, axis=1)
    for center_idx in range(n_obj):
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.argpartition(dis_vec, k_neig)[:k_neig]
        if center_idx not in nearest_idx:
            nearest_idx[-1] = center_idx  # Ensure the center_idx is included

        if is_probH:
            H[nearest_idx, center_idx] = np.exp(-dis_vec[nearest_idx] ** 2 / (m_prob * avg_dis[center_idx]) ** 2)
        else:
            H[nearest_idx, center_idx] = 1.0
    return H
'''
    n_obj = dis_mat.shape[0]
    H = np.zeros((n_obj, n_obj))

    for center_idx in range(n_obj):
        nearest_idx = np.argpartition(dis_mat[center_idx], k_neig)[:k_neig]
        if center_idx not in nearest_idx:
            nearest_idx[-1] = center_idx

        if is_probH:
            H[nearest_idx, center_idx] = np.exp(
                -dis_mat[center_idx, nearest_idx] ** 2 / (m_prob * np.mean(dis_mat[center_idx])) ** 2)
        else:
            H[nearest_idx, center_idx] = 1.0
    return H

def construct_H_with_KNN(X, y, K_neigs=[2], split_diff_scale=False, is_probH=True, m_prob=1):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    print(X.shape)
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    dis_mat = Eu_dis(X)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    print('No robuts!')
    #H= robust(X,y.astype(np.int32),H)
    #print('robuts!')
#     print(H)
    print(H.shape)
    return H

def construct_muiH_with_KNN(fts_info,fts_breakfast,fts_lunch,fts_dinner,fts_wg,fts_wljf,fts_library, fts_shopping,y, K_neigs=[2], split_diff_scale=False, is_probH=True, m_prob=1):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    #
    if len(fts_info.shape) != 2:
        fts_info = fts_info.reshape(-1, fts_info.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]
    dis_mat = Eu_dis(fts_info)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H_info = hyperedge_concat(H, H_tmp)
        else:
            H_info.append(H_tmp)
      #      
    if len(fts_breakfast.shape) != 2:
        fts_breakfast = fts_breakfast.reshape(-1, fts_breakfast.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]
    dis_mat = Eu_dis(fts_breakfast)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H_breakfast = hyperedge_concat(H, H_tmp)
        else:
            H_breakfast.append(H_tmp)
    #
    if len(fts_lunch.shape) != 2:
        fts_lunch = fts_lunch.reshape(-1, fts_lunch.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]
    dis_mat = Eu_dis(fts_lunch)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H_lunch = hyperedge_concat(H, H_tmp)
        else:
            H_lunch.append(H_tmp)
          #  
    if len(fts_dinner.shape) != 2:
        fts_dinner = fts_dinner.reshape(-1, fts_dinner.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]
    dis_mat = Eu_dis(fts_dinner)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H_dinner = hyperedge_concat(H, H_tmp)
        else:
            H_dinner.append(H_tmp)
    #
    if len(fts_wg.shape) != 2:
        fts_wg = fts_wg.reshape(-1, fts_wg.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]
    dis_mat = Eu_dis(fts_wg)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H_wg = hyperedge_concat(H, H_tmp)
        else:
            H_wg.append(H_tmp)
    #
    if len(fts_wljf.shape) != 2:
        fts_wljf = fts_wljf.reshape(-1, fts_wljf.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]
    dis_mat = Eu_dis(fts_wljf)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H_wljf = hyperedge_concat(H, H_tmp)
        else:
            H_wljf.append(H_tmp)
       #     
    if len(fts_library.shape) != 2:
        fts_library = fts_library.reshape(-1, fts_library.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]
    dis_mat = Eu_dis(fts_library)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H_library = hyperedge_concat(H, H_tmp)
        else:
            H_library.append(H_tmp)
       #     
#     if len(fts_shopping.shape) != 2:
#         fts_shopping = fts_shopping.reshape(-1, fts_shopping.shape[-1])

#     if type(K_neigs) == int:
#         K_neigs = [K_neigs]
#     dis_mat = Eu_dis(fts_shopping)
#     H = []
#     for k_neig in K_neigs:
#         H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
#         if not split_diff_scale:
#             H_shopping = hyperedge_concat(H, H_tmp)
#         else:
#             H_shopping.append(H_tmp)
            
            
    H = hyperedge_concat(H_info,H_breakfast,H_lunch,H_dinner,H_wg,H_wljf,H_library)
     
    return H

def construct_duibiH_with_KNN(fts_info,fts_breakfast,fts_lunch,fts_dinner,fts_wg,fts_wljf,fts_library,fts_shopping, y, K_neigs, split_diff_scale=False, is_probH=True, m_prob=1):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    #
    if len(fts_info.shape) != 2:
        fts_info = fts_info.reshape(-1, fts_info.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]
    dis_mat = Eu_dis(fts_info)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H_info = hyperedge_concat(H, H_tmp)
        else:
            H_info.append(H_tmp)
      #      
    if len(fts_breakfast.shape) != 2:
        fts_breakfast = fts_breakfast.reshape(-1, fts_breakfast.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]
    dis_mat = Eu_dis(fts_breakfast)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H_breakfast = hyperedge_concat(H, H_tmp)
        else:
            H_breakfast.append(H_tmp)
    #
    if len(fts_lunch.shape) != 2:
        fts_lunch = fts_lunch.reshape(-1, fts_lunch.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]
    dis_mat = Eu_dis(fts_lunch)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H_lunch = hyperedge_concat(H, H_tmp)
        else:
            H_lunch.append(H_tmp)
          #  
    if len(fts_dinner.shape) != 2:
        fts_dinner = fts_dinner.reshape(-1, fts_dinner.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]
    dis_mat = Eu_dis(fts_dinner)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H_dinner = hyperedge_concat(H, H_tmp)
        else:
            H_dinner.append(H_tmp)
    #
    if len(fts_wg.shape) != 2:
        fts_wg = fts_wg.reshape(-1, fts_wg.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]
    dis_mat = Eu_dis(fts_wg)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H_wg = hyperedge_concat(H, H_tmp)
        else:
            H_wg.append(H_tmp)
    #
    if len(fts_wljf.shape) != 2:
        fts_wljf = fts_wljf.reshape(-1, fts_wljf.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]
    dis_mat = Eu_dis(fts_wljf)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H_wljf = hyperedge_concat(H, H_tmp)
        else:
            H_wljf.append(H_tmp)
       #     
    if len(fts_library.shape) != 2:
        fts_library = fts_library.reshape(-1, fts_library.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]
    dis_mat = Eu_dis(fts_library)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H_library = hyperedge_concat(H, H_tmp)
        else:
            H_library.append(H_tmp)
       #     
    if len(fts_shopping.shape) != 2:
        fts_shopping = fts_shopping.reshape(-1, fts_shopping.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]
    dis_mat = Eu_dis(fts_shopping)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H_shopping = hyperedge_concat(H, H_tmp)
        else:
            H_shopping.append(H_tmp)
            
            
    H = hyperedge_concat(H_info,H_breakfast,H_lunch,H_dinner,H_wg,H_wljf,H_library,H_shopping)
     
    return H