#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/content/drive/My Drive/Colab Notebooks/HGDTI/data/',
                        help='path to data')
    parser.add_argument('--model_path', type=str, default='/content/drive/My Drive/Colab Notebooks/HGDTI/model_save/',
                        help='path to save model')
    parser.add_argument('--result_path', type=str, default='/content/drive/My Drive/Colab Notebooks/HGDTI/result/',
                        help='path to save result')
    parser.add_argument('--D_n', type=int, default=708,
                        help='number of drug node')
    parser.add_argument('--P_n', type=int, default=1512,
                        help='number of protein node')
    parser.add_argument('--S_n', type=int, default=4192,
                        help='number of side-effect node')
    parser.add_argument('--I_n', type=int, default=5603,
                        help='number of disease node')
    parser.add_argument('--embed_d', type=int, default=128,
                        help='embedding dimension')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--weight_decay', type=float, default=5e-3, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--patience', type=int, default=5, help='Patience')
    parser.add_argument('--filename', type=str, default='mat_drug_protein.txt',
                        help='dataset file')
    parser.add_argument('--train_iter_n', type=int, default=10,
                        help='max number of training iteration')
    parser.add_argument("--random_seed", default=10, type=int)
    parser.add_argument("--cuda", default=1, type=int)
    parser.add_argument("--t", default="o", type=str)
    parser.add_argument("--r", default="ten", type=str)
    parser.add_argument("--checkpoint", default='', type=str)

    args = parser.parse_args()

    return args
