import pandas as pd
import numpy as np
import csv

def load_fa(path):
    """a function to read fasta file from the path and store in a dict"""
    genes_seq = {}  #将序列存入字典
    with open(path,"r") as sequences:  #以读取方式打开文件
        lines = sequences.readlines()


    for line in lines:
        if line.startswith(">"):
            genename = line.split()[0]  #这个地方需要灵活调整
            genes_seq[genename] = ''  #序列为字符串
        else:
            genes_seq[genename] += line.strip()
    return genes_seq


def build_kmers(seq, k_size):
    """a function to calculate kmers from seq"""
    kmers = []  # k-mer存储在列表中
    n_kmers = len(seq) - k_size + 1

    for i in range(n_kmers):
        kmer = seq[i:i + k_size]
        kmers.append(kmer)

    return kmers

from collections import Counter
def summary_kmers(kmers):
    """a function to summarize the kmers"""
    kmers_stat = dict(Counter(kmers))
    return kmers_stat

genes_seq = load_fa(path="PHIAF/370h-PHIAF.fasta")
#genes_seq = load_fa(path="../data/PHIDetector/host.fasta")
genes_kmers = {}
for gene in genes_seq.keys():
  genes_kmers[gene] = summary_kmers(build_kmers(seq=genes_seq[gene], k_size=4))


kmers_stat_frame = pd.DataFrame(genes_kmers)


kmers_freq = lambda x : x/np.sum(x)
kmers_freq_frame =kmers_stat_frame.apply(kmers_freq, axis=0)
#print(kmers_freq_frame.shape)

# 1 筛选出所有名称中还有“A C G T”以外的数据记录
# kmers_freq_frame = kmers_freq_frame.drop(kmers_freq_frame[kmers_freq_frame.index.str.contains("R")].index)
all_index = kmers_freq_frame.index
delete_index = []
for str in all_index :
    for c in str :
        if not (c == 'A') | (c == 'T') | (c == 'C') | (c == 'G'):
            delete_index.append(str)
            break
kmers_freq_frame = kmers_freq_frame.drop(index=delete_index)
##转置每行为不同的phage/host
kmers_freq_frame = pd.DataFrame(kmers_freq_frame.values.T, index = kmers_freq_frame.columns, columns = kmers_freq_frame.index)

#print(kmers_freq_frame)
kmers_freq_frame.to_csv("PHIAF/370h-PHIAF_kmers3.csv",index = True, sep=',',header = True,na_rep='0')
#kmers_freq_frame.to_csv("../data/PHIDetector/host_kmers.csv",index = True, sep=',',header = True,na_rep='0')





