#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from urllib import parse,request
import requests
import numpy as np

__author__ = 'lion yu'


def read_drugs(filename, savename):
    drug_list = []
    for drug in open(filename, "r"):
        drug_list.append(drug.strip())
    drug_count = len(drug_list)
    sim_matrix = np.zeros((drug_count, drug_count))
    for i in range(drug_count):
        sim_matrix[i][i] = 1

    client = requests.session()
    headers = {'Connection': 'keep-alive'}
    for i in range(drug_count-1):
        query_1 = drug_list[i]
        query_2 = ""
        col_list = []
        p_c = 0
        no_url = 0
        for j in range(i+1, drug_count):
            query_2 += drug_list[j] + "+"
            p_c += 1
            if p_c % 500 == 0 or j + 1 == drug_count:
                no_url += 1
                url = 'http://rest.genome.jp/simcomp2/%s/%s/cutoff=%d' % (query_1, query_2.strip("+"), 0.0)
                print("no:", no_url, ", url:", url)
                req = client.get(url, headers=headers)
                res = req.content
                print(req.status_code)
                res = res.decode(encoding='utf-8').strip()
                results = list(map(m_l, res.split("\n")))
                col_list.extend(results)
                print("col_list:", len(col_list))
                query_2 = ""
                print("i_no:", i)
        # savefile = open(savename, "w")
        # savefile.write(str(col_list))
        # return
        print("i:", i, ", len:", len(col_list))
        sim_matrix[i, i+1:] = np.array(col_list)[:]
        sim_matrix[i+1:, i] = np.array(col_list)[:]
    np.savetxt(savename, sim_matrix, fmt='%.2e')


def m_l(line):
    # return line.split("\t")[1]
    return round(float(line.split("\t")[2]), 2)


if __name__ == "__main__":
    read_drugs("/content/drive/My Drive/Colab Notebooks/HGDTI/data/drug_list.txt", "/content/drive/My Drive/Colab Notebooks/HGDTI/data/sim_matrix.txt")