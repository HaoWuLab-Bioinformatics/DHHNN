"""Base model class."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score,accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.layers import FermiDiracDecoder
import manifolds
import models.encoders as encoders
from models.decoders import model2decoder
from utils.eval_utils import acc_f1
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve, auc, precision_score, confusion_matrix
'''
BaseModel类:

这是一个基础模型类，定义了图嵌入任务的基本结构和功能。
__init__ 方法初始化了模型的一些基本属性，如流形、编码器等。
encode 方法用于获取图的节点嵌入。
compute_metrics, init_metric_dict, has_improved 是抽象方法，需要在子类中具体实现。
'''
class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        #从参数中提取manifold的名称并保存为实例变量。
        self.manifold_name = args.manifold
        '''
        设置变量c。如果参数中提供了c的值，它将被转换为一个tensor，并根据是否使用CUDA移动到相应的设备上。
        如果没有提供c，它将被设置为一个可以学习的参数，初始值为1。
        '''
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        #动态地从manifolds模块中获取对应的manifold类，并实例化它。
        self.manifold = getattr(manifolds, self.manifold_name)()
        #如果使用的是Hyperboloid流形，特征维度将增加1。
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1
        #从参数中提取节点数量并保存为实例变量。
        self.nnodes = args.n_nodes
        #动态地从encoders模块中获取对应的encoder类，并用c和args实例化它。
        self.encoder = getattr(encoders, args.model)(self.c, args)
    #定义一个名为encode的方法，它接受输入x（特征）和adj（邻接矩阵）
    def encode(self, x, adj):
        #如果使用的是Hyperboloid流形，会在x的第一列插入一列全0。
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        #调用encoder的encode方法来获取嵌入。
        h = self.encoder.encode(x, adj)
        return h
    #定义一个抽象的compute_metrics方法，子类需要实现这个方法来计算特定任务的度量。
    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError

'''
NCModel类:

继承自BaseModel，用于节点分类任务。
__init__ 方法初始化了解码器和一些与节点分类相关的属性。
decode 方法用于将节点嵌入解码为分类输出。
compute_metrics 方法计算与节点分类相关的度量，如准确率和F1分数。
'''
class NCModel(BaseModel):
    """
    Base model for node classification task.
    """
    #__init__ 方法用于初始化对象。super(NCModel, self).__init__(args) 调用父类 BaseModel 的初始化方法。
    def __init__(self, args):
        super(NCModel, self).__init__(args)
        #从 model2decoder 字典中获取解码器类并初始化它。
        self.decoder = model2decoder[args.model](self.c, args)
        #根据类别的数量设置 F1 分数的平均方法。
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        #根据 pos_weight 参数设置类别权重。
        if args.pos_weight:
            self.weights = torch.Tensor([1., 1. / data['labels'][idx_train].mean()])
        else:
            self.weights = torch.Tensor([1.] * args.n_classes)
        #如果使用 CUDA，则将权重 Tensor 移动到 GPU。
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    #decode 方法用于解码嵌入。它接收嵌入 h、邻接矩阵 adj 和索引 idx，并返回对应的 softmax 输出。
    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return F.log_softmax(output[idx], dim=1)

    def compute_metrics(self, embeddings, data, split):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        loss = F.nll_loss(output, data['labels'][idx], self.weights)
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        # Move tensors to CPU if they are on GPU
        labels_cpu = data['labels'][idx].cpu()
        output_cpu = output.max(1)[1].cpu()

        # Calculate precision and recall
        precision = precision_score(labels_cpu, output_cpu, average=self.f1_average)
        recall = recall_score(labels_cpu, output_cpu, average=self.f1_average)

        metrics = {'loss': loss, 'acc': acc, 'f1': f1, 'precision': precision, 'recall': recall}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]

'''
LPModel类:

继承自BaseModel，用于链接预测任务。
__init__ 方法初始化了解码器和与链接预测相关的属性。
decode 方法用于将节点嵌入解码为链接预测的输出。
compute_metrics 方法计算与链接预测相关的度量，如ROC曲线下的面积和平均精度。
'''
class LPModel(BaseModel):
    """
    Base model for link prediction task.
    """
    #__init__ 方法用于初始化对象。super(LPModel, self).__init__(args) 调用父类 BaseModel 的初始化方法。
    def __init__(self, args):
        super(LPModel, self).__init__(args)
        #初始化一个 FermiDiracDecoder 对象。
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        # 从参数中获取真实边和假边的数量。
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges
    #decode 方法用于解码嵌入以进行链接预测。
    def decode(self, h, idx):
        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        probs = self.dc.forward(sqdist)
        return probs
    #compute_metrics 方法用于计算链接预测任务的度量。
    def compute_metrics(self, embeddings, data, split):
        if split == 'train':
            edges_false = data[f'{split}_edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)]
        else:
            edges_false = data[f'{split}_edges_false']
        pos_scores = self.decode(embeddings, data[f'{split}_edges'])
        neg_scores = self.decode(embeddings, edges_false)
        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        metrics = {'loss': loss, 'roc': roc, 'ap': ap}

        if split == 'test':
            # 计算基本指标
            f1 = f1_score(labels, np.round(preds))
            recall_test = recall_score(labels, np.round(preds))  # 重新命名以避免混淆
            accuracy = accuracy_score(labels, np.round(preds))

            # 计算 AUPR
            precision_curve, recall_curve, _ = precision_recall_curve(labels, preds)
            aupr = auc(recall_curve, precision_curve)

            # 计算特异性
            tn, fp, fn, tp = confusion_matrix(labels, np.round(preds)).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            # 计算精确率
            precision_score_val = precision_score(labels, np.round(preds))

            # 更新 metrics 字典
            metrics.update({'f1': f1, 'recall': recall_test, 'accuracy': accuracy,
                            'aupr': aupr, 'specificity': specificity, 'precision': precision_score_val})

            # 确保 loss 是单个数值
        if split != 'train':
            metrics['loss'] = metrics['loss'].item() if hasattr(metrics['loss'], "item") else metrics['loss']

        return metrics
    #init_metric_dict 方法用于初始化度量字典。
    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}
    #has_improved 方法用于比较两个度量字典，并判断模型性能是否有所改进。
    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])

