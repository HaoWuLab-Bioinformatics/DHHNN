from __future__ import division
from __future__ import print_function
import pandas as pd
import datetime
import json
import logging
import os
import pickle
import time
import re
import sklearn
import numpy as np
from sklearn.decomposition import PCA

import optimizers
import torch
from config import parser
from models.base_models import NCModel, LPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)


def visualize_embeddings(embeddings, labels, epoch=None):
    # 使用t-SNE对嵌入进行降维
    tsne = TSNE(n_components=2, random_state=0)
    embeddings_2d = tsne.fit_transform(embeddings)

    # 为不同的标签设置不同的颜色
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    if epoch is not None:
        plt.title(f't-SNE visualization of embeddings at Epoch {epoch}')
    else:
        plt.title('t-SNE visualization of embeddings')
    plt.show()

def train(args):
    def parse_metrics(metrics_str):
        # 使用正则表达式匹配键值对
        return {m.group(1): float(m.group(2)) for m in re.finditer(r'(\w+): (\d+\.\d+)', metrics_str)}

    '''
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    '''
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else  int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            models_dir = os.path.join(os.environ['LOG_DIR'], args.task, date)
            save_dir = get_dir_name(models_dir)
        else:
            save_dir = args.save_dir
        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                logging.StreamHandler()
                            ])

    logging.info(f'Using: {args.device}')
    #logging.info("Using seed {}.".format(args.seed))

    default_datapath = "./data"
    data_path = os.environ.get('DATAPATH', default_datapath)
    data = load_data(args, os.path.join(data_path, args.dataset))

    # Load data
    #data = load_data(args, os.path.join(os.environ['DATAPATH'], args.dataset))

    args.n_nodes, args.feat_dim = data['features'].shape
    if args.task == 'nc':
        Model = NCModel
        args.n_classes = int(data['labels'].max() + 1)
        logging.info(f'Num classes: {args.n_classes}')
    else:
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        if args.task == 'lp':
            Model = LPModel
        else:
            Model = RECModel
            # No validation for reconstruction task
            args.eval_freq = args.epochs + 1

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    # Model and optimizer
    model = Model(args)
    logging.info(str(model))
    optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info(f"Total number of parameters: {tot_params}")
    if args.cuda is not None and int(args.cuda) >= 0 :
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)
    # Train model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(data['features'], data['adj_train_norm'])
        train_metrics = model.compute_metrics(embeddings, data, 'train')
        train_metrics['loss'].backward()
        if args.grad_clip is not None:
            max_norm = float(args.grad_clip)
            all_params = list(model.parameters())
            for param in all_params:
                torch.nn.utils.clip_grad_norm_(param, max_norm)
        optimizer.step()
        lr_scheduler.step()
        if (epoch + 1) % args.log_freq == 0:
            logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                   'lr: {}'.format(lr_scheduler.get_last_lr()[0]),
                                   format_metrics(train_metrics, 'train'),
                                   'time: {:.4f}s'.format(time.time() - t)
                                   ]))
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            embeddings = model.encode(data['features'], data['adj_train_norm'])
            val_metrics = model.compute_metrics(embeddings, data, 'val')
            if (epoch + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))
            if model.has_improved(best_val_metrics, val_metrics):
                best_test_metrics = model.compute_metrics(embeddings, data, 'test')
                best_emb = embeddings.cpu()
                if args.save:
                    np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.detach().numpy())
                best_val_metrics = val_metrics
                counter = 0
            else:
                counter += 1
                if counter == args.patience and epoch > args.min_epochs:
                    logging.info("Early stopping")

                    '''
                    #隐空间可视化
                    with torch.no_grad():
                        model.eval()
                        embeddings = model.encode(data['features'], data['adj_train_norm'])  # 获取嵌入
                        labels = data['labels'].cpu().numpy()  # 获取标签
                        embeddings_cpu = embeddings.cpu().numpy()
                        silhouette_scores = sklearn.metrics.silhouette_score(embeddings_cpu, labels)#计算轮廓分数
                        visualize_embeddings(embeddings_cpu, labels)  # 可视化嵌入
                        print(silhouette_scores)



                    break
                    '''

                    with torch.no_grad():
                        model.eval()
                        embeddings = model.encode(data['features'], data['adj_train_norm'])  # 获取嵌入
                        labels = data['labels'].cpu().numpy()  # 获取标签
                        embeddings_cpu = embeddings.cpu().numpy()
                        silhouette_scores = sklearn.metrics.silhouette_score(embeddings_cpu, labels)  # 计算轮廓分数

                        # 使用PCA进行降维至3维
                        pca = PCA(n_components=3)
                        embeddings_pca = pca.fit_transform(embeddings_cpu)

                        # 可视化
                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection='3d')
                        ax.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], embeddings_pca[:, 2], c=labels,
                                   cmap='viridis')

                        plt.title('3D Embeddings Visualization')
                        plt.show()

                        print(silhouette_scores)

                    break








    logging.info("Optimization Finished!")
    logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    if not best_test_metrics:
        model.eval()
        best_emb = model.encode(data['features'], data['adj_train_norm'])
        best_test_metrics = model.compute_metrics(best_emb, data, 'test')
    logging.info(" ".join(["Val set results:", format_metrics(best_val_metrics, 'val')]))
    logging.info(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))



    excel_file = "training_results.xlsx"  # Excel文件名
    # 提取验证集和测试集的结果
    val_metrics = parse_metrics(format_metrics(best_val_metrics, 'val'))
    test_metrics = parse_metrics(format_metrics(best_test_metrics, 'test'))
    result = pd.DataFrame([{**{'epoch': i + 1}, **val_metrics, **test_metrics}])  # 创建新的DataFrame
    # 检查Excel文件是否存在并读取
    if os.path.exists(excel_file):
        df = pd.read_excel(excel_file)
        df = pd.concat([df, result], ignore_index=True)  # 合并DataFrame
    else:
        df = result
    # 保存更新后的DataFrame
    df.to_excel(excel_file, index=False)





    if args.save:
        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.cpu().detach().numpy())
        if hasattr(model.encoder, 'att_adj'):
            filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
            pickle.dump(model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
            print('Dumped attention adj: ' + filename)

        json.dump(vars(args), open(os.path.join(save_dir, 'config.json'), 'w'))
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
        logging.info(f"Saved model in {save_dir}")

if __name__ == '__main__':
    i=0
    while i<1:
        args = parser.parse_args()
        train(args)
        print('======================================================================================================================================================================')
        i=i+1
        print(i)

