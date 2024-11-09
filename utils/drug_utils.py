import numpy as np

import scipy.sparse as sp





def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj_ = sp.coo_matrix(adj)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_nomalized = adj_.dot(
        degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    adj_nomalized = adj_nomalized.tocoo()
    return sparse_to_tuple(adj_nomalized)


def constructNet(drug_dis_matrix):
    drug_matrix = np.matrix(
        np.zeros((drug_dis_matrix.shape[0], drug_dis_matrix.shape[0]), dtype=np.int8))
    dis_matrix = np.matrix(
        np.zeros((drug_dis_matrix.shape[1], drug_dis_matrix.shape[1]), dtype=np.int8))

    mat1 = np.hstack((drug_matrix, drug_dis_matrix))
    mat2 = np.hstack((drug_dis_matrix.T, dis_matrix))
    adj = np.vstack((mat1, mat2))
    return adj


def constructHNet(drug_dis_matrix, drug_matrix, dis_matrix):
    mat1 = np.hstack((drug_matrix, drug_dis_matrix))
    mat2 = np.hstack((drug_dis_matrix.T, dis_matrix))
    return np.vstack((mat1, mat2))
