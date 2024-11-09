"""Data utils functions for pre-processing and data loading."""
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
import pickle as pkl
import sys
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from hypergraph_utils import generate_G_from_H, construct_H_with_KNN,robust
import pandas as pd
import torch
import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
def load_data(args, datapath):
    if args.task == 'nc':
        data = load_data_nc(args.use_hygraph,args.dataset, args.use_feats, datapath, args.split_seed)
    else:
        data = load_data_lp(args.use_hygraph,args.dataset, args.use_feats, datapath)
        adj = data['adj_train']
        if args.task == 'lp':
            adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                    adj, args.val_prop, args.test_prop, args.split_seed
            )
            data['adj_train'] = adj_train
            data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
            data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
            data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
    data['adj_train_norm'], data['features'] = process(
            data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
    )
    if args.dataset == 'airport':
        data['features'] = augment(data['adj_train'], data['features'])
    return data


# ############### FEATURES PROCESSING ####################################


def process(adj, features, normalize_adj, normalize_feats):
    if isinstance(features, tuple):
        # 使用稀疏矩阵，而不是元组
        features = sp.csr_matrix((features[1], (features[0][:, 0], features[0][:, 1])), shape=features[2])

    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features

'''
def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
'''
def normalize(mx):
    rowsum = np.array(mx.sum(1))
    # Convert rowsum to a floating point type to avoid the ValueError
    rowsum = rowsum.astype(np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


'''
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
'''

from scipy.sparse import coo_matrix


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """将scipy稀疏矩阵转换为torch稀疏张量。"""

    # 检查sparse_mx是否已经是scipy.sparse矩阵类型
    if not isinstance(sparse_mx, (sp.csr_matrix, sp.csc_matrix, sp.coo_matrix)):
        # 如果不是，将numpy矩阵转换为COO格式的scipy稀疏矩阵
        sparse_mx = coo_matrix(sparse_mx)

    # 现在可以安全地转换为COO格式
    sparse_mx = sparse_mx.tocoo()

    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices, values, shape)


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


# ############### DATA SPLITS #####################################################


def mask_edges(adj, val_prop, test_prop, seed):
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false)  


def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


def compress_features(features, max_features=10000):
    if features.shape[1] > max_features:
        print(f"特征矩阵纬度 {features.shape[1]} 大于 {max_features}，正在进行压缩")
        # 使用 TruncatedSVD 替代 PCA
        svd = TruncatedSVD(n_components=max_features)
        features = svd.fit_transform(features)
    return features
# ############### LINK PREDICTION DATA LOADERS ####################################


def load_data_lp(use_hygraph,dataset, use_feats, data_path):
    if dataset in ['cora', 'pubmed']:
        adj, features = load_citation_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'disease_lp':
        adj, features = load_synthetic_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'airport':
        adj, features = load_data_airport(dataset, data_path, return_label=False)
    elif dataset in ['DrugBank_DDI', 'STRING_PPI','NDFRT_DDA','CTD_DDA']:
        adj, features =load_bionev_data_lp(dataset, use_feats, data_path)
    elif dataset =='SCMFDD-S':
        adj, features=load_SCMFDD(dataset, data_path)
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))

    if isinstance(features, tuple):
        # 使用稀疏矩阵本身，而不是元组
        features_matrix = sp.csr_matrix((features[1], (features[0][:, 0], features[0][:, 1])), shape=features[2])
    else:
        features_matrix = features

    if (use_hygraph == True):
        # 使用构建的特征矩阵x构建超图关联矩阵H
        H = construct_H_with_KNN(features_matrix, None, K_neigs=[50])  # 您可以根据需要调整K_neigs参数
        print("使用了超图结构作为特征")
        # HGNN+
        # H = torch.tensor(H)
        # G = Hypergraph.from_feature_kNN(H, 2)
        # 使用H生成特征矩阵G,HGNN
        G = generate_G_from_H(H)
        features = G

    data = {'adj_train': adj, 'features': features}
    return data


# ############### NODE CLASSIFICATION DATA LOADERS ####################################


def load_data_nc(use_hygraph,dataset, use_feats, data_path, split_seed):
    if dataset in ['cora', 'pubmed']:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
            dataset, use_feats, data_path, split_seed
        )
    elif dataset=='ppi':
        adj, features, labels, idx_train, idx_val, idx_test = load_PPI_data_simple(data_path
        )
    elif dataset in ['node2vec_PPI', 'Mashup_PPI']:
        adj, features, labels =load_bionev_data(
            dataset, use_feats, data_path, split_seed
        )
        val_prop, test_prop = 0.1, 0.1
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)
    elif dataset ==  'alzheimers':
        adj, features, labels = load_alzheimers_data(
            dataset, use_feats, data_path
        )
        val_prop, test_prop = 0.15, 0.15
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)

    elif dataset=='Clin_Term_COOC':
        adj, features, labels =load_bionev_data_simplified(
            dataset, use_feats, data_path, split_seed
        )
        val_prop, test_prop = 0.1, 0.1
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)
    elif dataset in ['diabet', 'diabetuci']:
        adj, features, labels =load_diabet(
            dataset, use_feats, data_path
        )
        val_prop, test_prop = 0.15, 0.15
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)
    elif dataset == 'TUDataset':
        val_prop, test_prop = 0.15, 0.15
        adj, features, labels, idx_train, idx_val, idx_test = load_PROTEINS_data(data_path, val_prop, test_prop, seed=split_seed)
    else:
        if dataset == 'disease_nc':
            adj, features, labels = load_synthetic_data(dataset, use_feats, data_path)
            val_prop, test_prop = 0.15, 0.15
        elif dataset == 'airport':
            adj, features, labels = load_data_airport(dataset, data_path, return_label=True)
            val_prop, test_prop = 0.15, 0.15
        else:
            raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)


    if(use_hygraph==True):
        # 使用构建的特征矩阵x构建超图关联矩阵H
        H = construct_H_with_KNN(features, labels, K_neigs=[5])  # 您可以根据需要调整K_neigs参数
        print("使用了超图结构作为特征")
        # HGNN+
        # H = torch.tensor(H)
        # G = Hypergraph.from_feature_kNN(H, 2)
        # 使用H生成特征矩阵G,HGNN
        G = generate_G_from_H(H)
        features = compress_features(G)



    labels = torch.LongTensor(labels)
    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}
    return data


# ############### DATASETS ####################################


def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + 500)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not use_feats:
        features = sp.eye(adj.shape[0])

    #以下为修改部分
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features, labels, idx_train, idx_val, idx_test


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_synthetic_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))


    return sp.csr_matrix(adj), features, labels


def load_data_airport(dataset_str, data_path, return_label=False):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    features = np.array([graph._node[u]['feat'] for u in graph.nodes()])
    if return_label:
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0/7, 8.0/7, 9.0/7])




        return sp.csr_matrix(adj), features, labels
    else:
        return sp.csr_matrix(adj), features




import os
import json
import numpy as np
from scipy.sparse import coo_matrix, block_diag

def convert_multilabel_to_single(labels):
    # 选择出现次数最多的标签
    single_labels = np.argmax(labels, axis=1)
    return single_labels

def load_PPI_data_simple(folder_path):
    folder_path=f"{folder_path}/raw"
    # Helper function to load numpy arrays
    def load_npy(file):
        return np.load(os.path.join(folder_path, file))

    # Helper function to load and process graph files
    def load_graph(file):
        with open(os.path.join(folder_path, file), 'r') as f:
            graph = json.load(f)
        nodes = [node['id'] for node in graph['nodes']]
        edge_list = [(edge['source'], edge['target']) for edge in graph['links']]
        row = [edge[0] for edge in edge_list]
        col = [edge[1] for edge in edge_list]
        data = np.ones(len(edge_list))
        adj = coo_matrix((data, (row, col)), shape=(len(nodes), len(nodes)))
        return adj

    # Load the data
    train_adj = load_graph('train_graph.json')
    val_adj = load_graph('valid_graph.json')
    test_adj = load_graph('test_graph.json')

    train_feats = load_npy('train_feats.npy')
    val_feats = load_npy('valid_feats.npy')
    test_feats = load_npy('test_feats.npy')

    train_labels = convert_multilabel_to_single(load_npy('train_labels.npy'))
    val_labels = convert_multilabel_to_single(load_npy('valid_labels.npy'))
    test_labels = convert_multilabel_to_single(load_npy('test_labels.npy'))

    # Concatenate the data
    adj = block_diag((train_adj, val_adj, test_adj))
    features = np.vstack((train_feats, val_feats, test_feats))
    labels = np.concatenate((train_labels, val_labels, test_labels))

    # Create index arrays
    idx_train = np.arange(train_feats.shape[0])
    idx_val = np.arange(train_feats.shape[0], train_feats.shape[0] + val_feats.shape[0])
    idx_test = np.arange(train_feats.shape[0] + val_feats.shape[0],
                         train_feats.shape[0] + val_feats.shape[0] + test_feats.shape[0])


    return adj, features, labels, idx_train, idx_val, idx_test

import numpy as np
import scipy.sparse as sp
import os
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder


def load_bionev_data(dataset_str, use_feats, data_path, split_seed=None):
    # 构造文件路径
    edgelist_file_path = os.path.join(data_path, f'{dataset_str}.edgelist')
    labels_file_path = os.path.join(data_path, f'{dataset_str}_labels.txt')

    # 读取边并构建邻接矩阵
    edges = np.loadtxt(edgelist_file_path, dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(np.max(edges) + 1, np.max(edges) + 1))

    # 根据 use_feats 参数构建特征矩阵
    if use_feats:
        features = sp.identity(np.max(edges) + 1)
    else:
        # 如果不使用特征，则创建一个单位特征矩阵
        features = np.ones((adj.shape[0], 1))

    # 使用 read_node_labels 函数读取标签
    def read_node_labels(filename):
        fin = open(filename, 'r')
        node_list = []
        labels = []
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.strip().split()
            node_list.append(int(vec[0]))  # 节点ID不减一
            labels.append(int(vec[1]))
        fin.close()
        return node_list, labels

    node_list, labels_list = read_node_labels(labels_file_path)
    labels = np.zeros((features.shape[0],))
    labels[node_list] = labels_list

    return adj, features, labels


import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder


def load_bionev_data_simplified(dataset_str, use_feats, data_path, split_seed=None):
    # 构造文件路径
    edgelist_file_path = os.path.join(data_path, f'{dataset_str}.edgelist')
    labels_file_path = os.path.join(data_path, f'{dataset_str}_labels.txt')
    node_list_file_path = os.path.join(data_path, 'node_list.txt')

    # 读取边并构建邻接矩阵
    edges = np.loadtxt(edgelist_file_path, dtype=np.float32)
    adj = sp.coo_matrix((edges[:, 2], (edges[:, 0], edges[:, 1])),
                        shape=(int(np.max(edges[:, 0:2])) + 1, int(np.max(edges[:, 0:2])) + 1))

    # 读取节点列表
    nodes = np.loadtxt(node_list_file_path, dtype=int, skiprows=1, usecols=(0,))  # 跳过第一行（标题行）并只读取第一列

    # 根据 use_feats 参数构建特征矩阵
    if use_feats:
        features = sp.identity(np.max(nodes) + 1)
    else:
        # 如果不使用特征，则创建一个单位特征矩阵
        features = np.ones((adj.shape[0], 1))

    # 使用 read_node_labels 函数读取标签
    def read_node_labels(filename):
        fin = open(filename, 'r')
        next(fin)  # 跳过标题行
        node_list = []
        labels = []
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.strip().split()
            node_list.append(int(vec[0]))  # 节点ID不减一
            labels.append(int(vec[1]))
        fin.close()
        return node_list, labels

    node_list, labels_list = read_node_labels(labels_file_path)
    labels = np.zeros((features.shape[0],))
    labels[node_list] = labels_list

    return adj, features, labels


import numpy as np
import scipy.sparse as sp
import os

def load_bionev_data_lp(dataset_str, use_feats, data_path, split_seed=None):
    # 构造文件路径
    edgelist_file_path = os.path.join(data_path, f'{dataset_str}.edgelist')
    node_list_file_path = os.path.join(data_path, 'node_list.txt')

    # 读取边并构建邻接矩阵
    edges = np.loadtxt(edgelist_file_path, dtype=np.float32)
    # 检查边列表是否包含权重
    if edges.shape[1] == 2:
        # 无权图，所有边的权重设置为1
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(int(np.max(edges[:, 0:2])) + 1, int(np.max(edges[:, 0:2])) + 1))
    elif edges.shape[1] == 3:
        # 带权图
        adj = sp.coo_matrix((edges[:, 2], (edges[:, 0], edges[:, 1])),
                            shape=(int(np.max(edges[:, 0:2])) + 1, int(np.max(edges[:, 0:2])) + 1))
    else:
        raise ValueError("边列表的格式不正确。应该有2或3列。")

    # 读取节点列表
    nodes = np.loadtxt(node_list_file_path, dtype=int, skiprows=1, usecols=(0,))  # 跳过第一行（标题行）并只读取第一列

    # 根据 use_feats 参数构建特征矩阵
    if use_feats:
        features = sp.identity(np.max(nodes) + 1)
    else:
        # 如果不使用特征，则创建一个单位特征矩阵
        features = np.ones((adj.shape[0], 1))

    return adj, features

from drug_utils import constructHNet,constructNet,sparse_to_tuple
def load_SCMFDD(dataset, data_path, split_seed=None):
    edgelist_file_path = os.path.join(data_path, 'drugsim.csv')
    labels_file_path = os.path.join(data_path, 'dissim.csv')
    node_list_file_path = os.path.join(data_path, 'drug_disease_sim.csv')
    drug_sim = np.loadtxt(edgelist_file_path, delimiter=',')
    dis_sim = np.loadtxt(labels_file_path, delimiter=',')
    drug_dis_matrix = np.loadtxt(node_list_file_path, delimiter=',')
    adj = constructHNet(drug_dis_matrix, drug_sim, dis_sim)
    adj = sp.csr_matrix(adj)
    #association_nam = drug_dis_matrix.sum()
    features = sparse_to_tuple(sp.csr_matrix(constructNet(drug_dis_matrix)))
    # 根据 use_feats 参数构建特征矩阵
    return adj,features




def load_diabet(dataset, use_feats, data_path):
    # 构造文件路径
    data_file_path = os.path.join(data_path, f'{dataset}.csv')

    # 读取数据
    data = pd.read_csv(data_file_path)

    # 如果使用特征，则特征矩阵为除了最后一列的所有列
    if use_feats:
        feature_values = data.iloc[:, :-1].values
        features = sp.csr_matrix(feature_values)
    else:
        # 如果不使用特征，则创建一个单位特征矩阵
        features = np.ones((data.shape[0], 1))

    # 标签为最后一列
    labels = data.iloc[:, -1].values

    # 使用KNN构建邻接矩阵，固定使用 5 个邻居
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(feature_values)
    knn_distances, knn_indices = knn.kneighbors(feature_values)

    # 初始化邻接矩阵
    adj = sp.lil_matrix((data.shape[0], data.shape[0]))

    # 填充邻接矩阵
    for i in range(data.shape[0]):
        for j in knn_indices[i]:
            adj[i, j] = 1
            adj[j, i] = 1  # 确保邻接矩阵是对称的

    # 转换为CSR格式
    adj = adj.tocsr()

    return adj, features, labels

import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

def load_alzheimers_data(dataset, use_feats, data_path):
    # 构造文件路径
    #data_file_path = os.path.join(data_path, f'{dataset}.csv')
    data_file_path = os.path.join(data_path, f'{dataset}.xlsx')

    # 读取数据
    data= pd.read_excel(data_file_path)
    #data = pd.read_csv(data_file_path)

    # 确定标签列 ('Group') 的索引并将其作为标签
    label_col = 'Group'
    label_index = data.columns.get_loc(label_col)

    # 特征为除了 'Group' 列的所有列
    if use_feats:
        feature_values = data.drop(label_col, axis=1).values
        features = sp.csr_matrix(feature_values)
    else:
        # 如果不使用特征，则创建一个单位特征矩阵
        features = np.ones((data.shape[0], 1))

    # 标签为 'Group' 列
    labels = data[label_col].values

    # 使用KNN构建邻接矩阵，固定使用 5 个邻居
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(feature_values)
    knn_distances, knn_indices = knn.kneighbors(feature_values)

    # 初始化邻接矩阵
    adj = sp.lil_matrix((data.shape[0], data.shape[0]))

    # 填充邻接矩阵
    for i in range(data.shape[0]):
        for j in knn_indices[i]:
            adj[i, j] = 1
            adj[j, i] = 1  # 确保邻接矩阵是对称的

    # 转换为CSR格式
    adj = adj.tocsr()

    return adj, features, labels


import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp

def flatten_graph(adj, features, labels):
    """将图的邻接矩阵和特征矩阵展平"""
    num_nodes = features.shape[0]
    flattened_features = features.flatten()
    flattened_labels = np.zeros(num_nodes + flattened_features.shape[0])
    flattened_labels[:num_nodes] = labels.numpy()
    flattened_labels[num_nodes:] = flattened_features
    return flattened_labels

from sklearn.model_selection import train_test_split


def load_PROTEINS_data(data_path, val_prop, test_prop, seed=None):
    # 加载 PROTEINS 数据集
    dataset = TUDataset(root=data_path, name='PROTEINS')

    # 初始化列表用于存储邻接矩阵、特征和标签
    adj_list = []
    features_list = []
    labels_list = []

    # 找到数据集中所有图的最大特征维度
    max_feature_dim = max(data.x.shape[1] for data in dataset)

    # 记录每个图的节点数量，用于后续构建块对角矩阵
    num_nodes_list = []

    # 遍历数据集，提取每个图的邻接矩阵、特征和标签
    for data in dataset:
        edge_index = data.edge_index
        x = data.x
        y = data.y

        # 如果当前图的特征维度小于最大特征维度，则进行填充
        if x.shape[1] < max_feature_dim:
            padding = torch.zeros((x.shape[0], max_feature_dim - x.shape[1]))
            x = torch.cat([x, padding], dim=1)

        # 构建邻接矩阵
        num_nodes = x.size(0)
        num_nodes_list.append(num_nodes)
        adj = sp.coo_matrix(
            (np.ones(edge_index.size(1)), (edge_index[0].numpy(), edge_index[1].numpy())),
            shape=(num_nodes, num_nodes)
        )

        # 将特征转换为 NumPy 数组
        features_np = x.numpy() if isinstance(x, torch.Tensor) else np.array(x)

        # 将邻接矩阵、特征和标签添加到对应的列表中
        adj_list.append(adj.tocsr())  # 转换为 csr_matrix 格式
        features_list.append(features_np)
        labels_list.append(y.item())

    # 整合邻接矩阵为块对角矩阵
    adj = sp.block_diag(adj_list)

    # 整合特征矩阵
    features = np.vstack(features_list)

    # 将标签列表转换为 NumPy 数组
    labels = np.array(labels_list)

    # 设置随机种子
    np.random.seed(seed)

    # 划分数据集索引
    indices = np.arange(len(labels))
    idx_train, idx_temp = train_test_split(indices, test_size=(val_prop + test_prop), random_state=seed)
    idx_val, idx_test = train_test_split(idx_temp, test_size=test_prop / (val_prop + test_prop), random_state=seed)

    return adj, sp.csr_matrix(features), labels, idx_train, idx_val, idx_test
