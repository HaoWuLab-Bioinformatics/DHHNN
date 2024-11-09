#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
import tools
from args import read_args
import numpy as np
import math
import random
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef, recall_score, average_precision_score, \
    confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

torch.set_num_threads(2)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class model_class(object):
    def __init__(self, args):
        super(model_class, self).__init__()
        self.args = args
        self.gpu = args.cuda
        self.seed = np.random.randint(0, 1000, args.train_iter_n)
        print("seed:", self.seed)
        network_path = self.args.data_path
        true_drug = 708
        # self.drug_protein = np.loadtxt(network_path + 'mat_drug_protein.txt')
        self.drug_drug = np.loadtxt(network_path + 'mat_drug_drug.txt')
        drug_chemical = np.loadtxt(network_path + 'Similarity_Matrix_Drugs.txt')
        self.drug_chemical = drug_chemical[:true_drug, :true_drug]
        self.drug_disease = np.loadtxt(network_path + 'mat_drug_disease.txt')
        self.drug_sideeffect = np.loadtxt(network_path + 'mat_drug_se.txt')
        self.protein_protein = np.loadtxt(network_path + 'mat_protein_protein.txt')
        self.protein_sequence = np.loadtxt(network_path + 'Similarity_Matrix_Proteins.txt')
        self.protein_disease = np.loadtxt(network_path + 'mat_protein_disease.txt')
        self.drug_matrix = self.combine_matrixs(self.drug_drug, self.drug_chemical)
        self.protein_matrix = self.combine_matrixs(self.protein_protein, self.protein_sequence)

        self.drug_drug_normalize = torch.from_numpy(self.row_normalize(self.drug_drug, True)).float()
        self.drug_chemical_normalize = torch.from_numpy(self.row_normalize(self.drug_chemical, True)).float()
        self.drug_disease_normalize = torch.from_numpy(self.row_normalize(self.drug_disease, False)).float()
        self.drug_sideeffect_normalize = torch.from_numpy(self.row_normalize(self.drug_sideeffect, False)).float()

        self.protein_protein_normalize = torch.from_numpy(self.row_normalize(self.protein_protein, True)).float()
        self.protein_sequence_normalize = torch.from_numpy(self.row_normalize(self.protein_sequence, True)).float()
        self.protein_disease_normalize = torch.from_numpy(self.row_normalize(self.protein_disease, False)).float()

        self.model = tools.HetAgg(args, self.drug_drug_normalize, self.drug_chemical_normalize,
                                  self.drug_disease_normalize, self.drug_sideeffect_normalize,
                                  self.protein_protein_normalize, self.protein_sequence_normalize,
                                  self.protein_disease_normalize)

        self.model.init_weights()
        if self.gpu:
            self.model.cuda()
        self.parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optim = optim.Adam(self.parameters, lr=self.args.lr, weight_decay=0)
        

    def model_train(self):
        print('model training ...')
        if self.args.checkpoint != '':
            print("model state loading.....")
            self.model.load_state_dict(torch.load(self.args.checkpoint))

        self.model.train()
        whole_positive_index, whole_negative_index, negative_sample, whole_positive_index_test, whole_negative_index_test, negative_sample_test = self.sample_neg_links(0.5)
        
        auc_round = []
        aupr_round = []
        mcc_round = []
        sp_round = []
        sn_round = []
        rs = np.random.randint(0, 1000, 1)[0]
        for iter_i in range(self.args.train_iter_n):
            print('iteration ' + str(iter_i + 1) + ' ...')
            data_set, data_set_test = self.sample_links(iter_i, whole_positive_index, whole_negative_index, negative_sample, whole_positive_index_test, whole_negative_index_test, negative_sample_test)
            print("sample_links success")

            test_auc_round = 0
            test_aupr_round = 0
            test_mcc_round = 0
            test_sp_round = 0
            test_sn_round = 0
            step_r = int(math.pow(2/3, iter_i) * self.args.loop)
            if step_r < 10:
                step_r = 10
            print("loop:", step_r)
            for step in range(step_r):
                print('epoch step', iter_i + 1, step + 1)
                if self.args.t == 'unique':
                    DTItrain = data_set
                    DTItest = data_set_test
                    DTItrain, DTIvalid = train_test_split(DTItrain, test_size=0.05, random_state=rs)
                    v_auc, v_aupr, t_auc, t_aupr, t_mcc, t_sp, t_sn = self.train_and_evaluate(DTItrain=DTItrain,
                                                                                              DTIvalid=DTIvalid,
                                                                                              DTItest=DTItest, num_step=0)
                    if t_aupr > test_aupr_round:
                        test_auc_round = t_auc
                        test_aupr_round = t_aupr
                        test_mcc_round = t_mcc
                        test_sp_round = t_sp
                        test_sn_round = t_sn
                else:
                    test_auc_fold = []
                    test_aupr_fold = []
                    test_mcc_fold = []
                    test_sp_fold = []
                    test_sn_fold = []
                    kf = StratifiedKFold(n_splits=10, shuffle=False)
                    step = 1
                    for train_index, test_index in kf.split(data_set[:, :2], data_set[:, 2]):
                        DTItrain, DTItest = data_set[train_index], data_set[test_index]
                        if self.args.r == 'ten':
                            DTItrain, DTIvalid = train_test_split(DTItrain, test_size=0.05, random_state=rs)
                        else:
                            DTIvalid = []
                        v_auc, v_aupr, t_auc, t_aupr, t_mcc, t_sp, t_sn = self.train_and_evaluate(DTItrain=DTItrain,
                                                                                                  DTIvalid=DTIvalid,
                                                                                                  DTItest=DTItest,
                                                                                                  num_step=step)

                        test_auc_fold.append(t_auc)
                        test_aupr_fold.append(t_aupr)
                        test_mcc_fold.append(t_mcc)
                        test_sp_fold.append(t_sp)
                        test_sn_fold.append(t_sn)
                        step += 1
                    print("aupr_mean:", np.mean(test_aupr_fold))
                    if np.mean(test_aupr_fold) > test_aupr_round:
                        test_auc_round = np.mean(test_auc_fold)
                        test_aupr_round = np.mean(test_aupr_fold)
                        test_mcc_round = np.mean(test_mcc_fold)
                        test_sp_round = np.mean(test_sp_fold)
                        test_sn_round = np.mean(test_sn_fold)

            auc_round.append(test_auc_round)
            aupr_round.append(test_aupr_round)
            mcc_round.append(test_mcc_round)
            sp_round.append(test_sp_round)
            sn_round.append(test_sn_round)
            print('epoch auc aupr mcc sp sn', iter_i + 1, test_auc_round, test_aupr_round,
                  test_mcc_round, test_sp_round, test_sn_round)
            np.savetxt(self.args.result_path + 'test_auc' + str(iter_i + 1), auc_round)
            np.savetxt(self.args.result_path + 'test_aupr' + str(iter_i + 1), aupr_round)
            np.savetxt(self.args.result_path + 'test_mcc' + str(iter_i + 1), mcc_round)
            np.savetxt(self.args.result_path + 'test_sp' + str(iter_i + 1), sp_round)
            np.savetxt(self.args.result_path + 'test_sn' + str(iter_i + 1), sn_round)
            print('end result auc aupr', np.mean(auc_round), np.mean(aupr_round))
            torch.save(self.model.state_dict(), self.args.model_path + "HetGNN_" + str(iter_i) + ".pt")

    def train_and_evaluate(self, DTItrain, DTIvalid, DTItest, num_step):
        best_valid_aupr = 0
        best_valid_auc = 0
        test_aupr = 0
        test_auc = 0
        test_mcc = 0
        drug_protein_valid, protein_drug_valid = self.DTI_normalize(DTIvalid)
        drug_protein_test, protein_drug_test = self.DTI_normalize(DTItest)
        drug_protein_train, protein_drug_train = self.DTI_normalize(DTItrain)
        outputs = self.model(DTItrain, drug_protein_train, protein_drug_train)
        loss = tools.binary_cross_entropy_loss(self.gpu, outputs, torch.from_numpy(DTItrain[:, 2]).float())
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        # print('fold', num_step, 'total_loss', loss.item())

        with torch.no_grad():
            v_outputs = self.model(DTIvalid, drug_protein_valid, protein_drug_valid).cpu().numpy()
            valid_auc = roc_auc_score(DTIvalid[:, 2], v_outputs)
            valid_aupr = average_precision_score(DTIvalid[:, 2], v_outputs)
            best_valid_aupr = valid_aupr
            best_valid_auc = valid_auc
            t_outputs = self.model(DTItest, drug_protein_test, protein_drug_test).cpu().numpy()
            y_predict_class = t_outputs
            y_predict_class[y_predict_class > 0.5] = 1
            y_predict_class[y_predict_class <= 0.5] = 0
            test_auc = roc_auc_score(DTItest[:, 2], t_outputs)
            test_aupr = average_precision_score(DTItest[:, 2], t_outputs)
            test_mcc = matthews_corrcoef(DTItest[:, 2], y_predict_class)
            test_matrix = confusion_matrix(DTItest[:, 2], y_predict_class)
            TP = test_matrix[1, 1]
            FP = test_matrix[0, 1]
            FN = test_matrix[1, 0]
            TN = test_matrix[0, 0]
            test_sp = TP / (TP + FN)
            test_sn = TN / (TN + FP)
            print('fold', num_step, 'valid auc aupr,', best_valid_auc, best_valid_aupr, 'test auc aupr mcc sp sn', test_auc, test_aupr, test_mcc,
                  test_sp, test_sn)
        return best_valid_auc, best_valid_aupr, test_auc, test_aupr, test_mcc, test_sp, test_sn

    def DTI_normalize(self, DTIData):
        drug_protein = np.zeros((self.args.D_n, self.args.P_n))
        for ele in DTIData:
            drug_protein[ele[0], ele[1]] = ele[2]
        protein_drug = drug_protein.T
        drug_protein_normalize = torch.from_numpy(self.row_normalize(drug_protein, False)).float()
        protein_drug_normalize = torch.from_numpy(self.row_normalize(protein_drug, False)).float()
        return drug_protein_normalize, protein_drug_normalize

    def row_normalize(self, a_matrix, substract_self_loop):
        if substract_self_loop == True:
            np.fill_diagonal(a_matrix, 0)
        a_matrix = a_matrix.astype(float)
        row_sums = a_matrix.sum(axis=1) + 1e-12
        new_matrix = a_matrix / row_sums[:, np.newaxis]
        new_matrix[np.isnan(new_matrix) | np.isinf(new_matrix)] = 0.0
        return new_matrix

    def compute_similar(self, list, id, matrix):
        sum = 0
        for index in list:
            sum += matrix[index][id]
        return sum

    def combine_matrixs(self, matrix, matrix_si):
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i == j:
                    continue
                weight = float(matrix[i][j])
                similar = float(matrix_si[i][j])
                weight += similar
                matrix[i][j] = weight
        return matrix

    def sample_neg_links(self, ratio):
        print("sampling ...")
        D_n = self.args.D_n
        P_n = self.args.P_n
        args = self.args
        if args.t == 'o':
            dti_o = np.loadtxt(args.data_path + args.filename)
        else:
            dti_o = np.loadtxt(args.data_path + 'mat_drug_protein_' + args.t + '.txt')
            ratio = 0.3
            unique_ratio = 0.7

        whole_positive_index = []
        whole_negative_index = []
        whole_positive_index_test = []
        whole_negative_index_test = []
        for i in range(D_n):
            for j in range(P_n):
                if int(dti_o[i][j]) == 1:
                    whole_positive_index.append([i, j])
                elif int(dti_o[i][j]) == 0:
                    whole_negative_index.append([i, j])
                if args.t == 'unique':
                    if int(dti_o[i][j]) == 3:
                        whole_positive_index_test.append([i, j])
                    elif int(dti_o[i][j]) == 2:
                        whole_negative_index_test.append([i, j])
        
        d_p_list = [[] for i in range(D_n)]
        p_d_list = [[] for i in range(P_n)]
        similar_matrix = []
        similar_matrix_test = []
        for i in range(D_n):
            for j in range(P_n):
                if dti_o[i][j] == 1:
                    d_p_list[i].append(j)
                    p_d_list[j].append(i)
        for i in range(D_n):
            for j in range(P_n):
                if dti_o[i][j] == 0 or dti_o[i][j] == 2:
                    d_list = p_d_list[j]
                    p_list = d_p_list[i]
                    sum = self.compute_similar(p_list, j, self.protein_sequence)
                    sum += self.compute_similar(d_list, i, self.drug_chemical)
                    if dti_o[i][j] == 0:
                        similar_matrix.append([math.exp(-sum), i, j])
                    else:
                        similar_matrix_test.append([math.exp(-sum), i, j])
        similar_matrix.sort(key=lambda item: item[0], reverse=True)
        # np.savetxt(self.args.data_path + 'similar_matrix.txt', np.array(similar_matrix)[:, 0])
        negative_size = int(len(similar_matrix) * ratio)
        negative_sample = np.array(similar_matrix)[:negative_size, 1:]
        negative_sample_test = []
        if args.t == 'unique':
            print('ratio:', unique_ratio)
            similar_matrix_test.sort(key=lambda item: item[0], reverse=True)
            negative_size_test = int(len(similar_matrix_test) * unique_ratio)
            negative_sample_test = np.array(similar_matrix_test)[:negative_size_test, 1:]
            # np.savetxt(self.args.data_path + 'similar_matrix_test.txt', np.array(similar_matrix_test)[:, 0])
        return whole_positive_index, np.array(whole_negative_index), negative_sample, whole_positive_index_test, np.array(whole_negative_index_test), negative_sample_test

    def sample_links(self, iter_i, whole_positive_index, whole_negative_index, negative_sample, whole_positive_index_test, whole_negative_index_test, negative_sample_test):
        np.random.seed(self.seed[iter_i])
        if args.r == 'ten':
            train_index = np.random.choice(np.arange(len(negative_sample)), size=10 * len(whole_positive_index), replace=False)
            # print("train_index:", train_index[:10])
            negative_sample_index = negative_sample[train_index]
        elif args.r == 'all':
            negative_sample_index = whole_negative_index
        else:
            print('wrong positive negative ratio')

        print("positive size:", len(whole_positive_index), ", negative size:", len(negative_sample_index))

        data_set = np.zeros((len(whole_positive_index) + len(negative_sample_index), 3), dtype=int)
        count = 0
        for i in whole_positive_index:
            data_set[count][0] = i[0]
            data_set[count][1] = i[1]
            data_set[count][2] = 1
            count += 1
        for i in negative_sample_index:
            data_set[count][0] = i[0]
            data_set[count][1] = i[1]
            data_set[count][2] = 0
            count += 1

        data_set_test = []
        if args.t == 'unique':
            if args.r == 'ten':
                test_index = np.random.choice(np.arange(len(negative_sample_test)), size=10 * len(whole_positive_index_test), replace=False)
                negative_sample_index_test = negative_sample_test[test_index]
            elif args.r == 'all':
                negative_sample_index_test = whole_negative_index_test
            else:
                print('wrong positive negative ratio')

            print("unique positive size:", len(whole_positive_index_test), ", unique negative size:", len(negative_sample_index_test))

            data_set_test = np.zeros((len(negative_sample_index_test) + len(whole_positive_index_test), 3), dtype=int)
            count = 0
            for i in whole_positive_index_test:
                data_set_test[count][0] = i[0]
                data_set_test[count][1] = i[1]
                data_set_test[count][2] = 1
                count += 1
            for i in negative_sample_index_test:
                data_set_test[count][0] = i[0]
                data_set_test[count][1] = i[1]
                data_set_test[count][2] = 0
                count += 1

        return data_set, data_set_test


if __name__ == '__main__':
    args = read_args()
    print("------arguments-------")
    for k, v in vars(args).items():
        print(k + ': ' + str(v))

    # fix random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    # model
    model_object = model_class(args)
    model_object.model_train()
