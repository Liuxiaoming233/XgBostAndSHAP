import numpy as np
import pandas as pd
from babel.dates import date_


def generate_all_kmers():
    """
    生成所有可能的k-mer组合。

    参数：
    k (int): k-mer的长度。

    返回：
    set: 所有可能的k-mer组合。
    """
    bases = 'ARNDCQEGHILKMFPSTWYV'
    kmer = []
    for l1 in bases:
        for l2 in bases:
            kmer.append(l1+l2)
    return kmer

def get2mer():
    keys = generate_all_kmers()
    data = open('sdandsar_potein/twomer/z_scores.txt')
    dataList = []
    for i in data:
        data_i = np.zeros(20)
        num = 0
        for x in i.strip('\n').strip(' ').split(' '):
            if x != '':
                data_i[num] = x
                num += 1
        dataList.append(np.array(data_i))
        # 转化为方阵
        B = np.tril(dataList) + np.tril(dataList, -1).T
        return B.reshape(1,400)
        # DF = pd.DataFrame(columns=keys,data=B.reshape(1,400),index=["z_scores"])
        # DF.to_csv('stander.csv')
def get1mer():
    keys = 'ARNDCQEGHILKMFPSTWYV'
    data = open('./sdandsar_potein/onemer/Molecular_weight.txt','r')
    data = [x for x in data]
    aaValuesL1 = []
    for i in data[1].strip('\n').split(' '):
        if i != '':
            aaValuesL1.append(float(i))
    aaValuesL2 = []
    for i in data[2].strip('\n').split(' '):
        if i != '':
            aaValuesL2.append(float(i))
    print(aaValuesL1)
    print(aaValuesL2)
    aaDict = {}
    num = 0
    for i in data[0].strip('\n').split(' '):
        if i != '':
            aaDict[i[0]] = aaValuesL1[num]
            aaDict[i[1]] = aaValuesL2[num]
            num += 1
    return [aaDict[x] for x in keys]

def getIndex1():
    data = open('./sdandsar_potein/AAindex1.txt','r')
    print(data)
getIndex1()
