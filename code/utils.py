import numpy as np
#import torch
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import scipy.sparse as sp
from tensorflow.python.ops import array_ops


def weight_variable_glorot(input_dim,output_dim,name=""):
    init_range = np.sqrt(6.0/(input_dim+output_dim))
    initial = tf.compat.v1.random_uniform(
        [input_dim,output_dim],
        minval=-init_range,
        maxval=init_range,
        dtype=tf.float32
    )

    return tf.Variable(initial,name=name)


def dropout_sparse(x,keep_prob,num_nonzero_elems):
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor +=tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor),dtype=tf.bool)
    pre_out = tf.sparse_retain(x,dropout_mask)
    return pre_out*(1./keep_prob)

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row,sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords,values,shape

def preprocess_graph(adj):
    adj_ = sp.coo_matrix(adj)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum,-0.5).flatten())
    adj_nomalized = adj_.dot(
        degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    adj_nomalized = adj_nomalized.tocoo()
    return  sparse_to_tuple(adj_nomalized)

def constructNet(met_dis_matrix):
    met_matrix = np.matrix(
        np.zeros((met_dis_matrix.shape[0],met_dis_matrix.shape[0]),dtype=np.int8))
    dis_matrix = np.matrix(
        np.zeros((met_dis_matrix.shape[1],met_dis_matrix.shape[1]),dtype=np.int8))

    mat1 = np.hstack((met_matrix,met_dis_matrix))
    mat2 = np.hstack((met_dis_matrix.T,dis_matrix))
    adj = np.vstack((mat1,mat2))
    return adj

def constructHNet(met_dis_matrix,met_matrix,dis_matrix):
    mat1 = np.hstack((met_matrix,met_dis_matrix))
    mat2 = np.hstack((met_dis_matrix.T,dis_matrix))
    return np.vstack((mat1,mat2))



def constructHNetPHP(train_met_dis_matrix, met_matrix):
    # 计算矩阵的乘积
    result = np.dot(train_met_dis_matrix.T, met_matrix)
    #result = np.dot(result, met_matrix)
    #processed_result = np.where(result > 0, 1, 0)

    return result.T



def merge(A, B):
    C = np.maximum(A, B)
    return C
    
def KNN(W, K):
    m, n = W.shape
    DS = np.zeros((m, n))
    for i in range(m):
        index = np.argsort(W[i, :])[-K:]
        DS[i, index] = W[i, index]
    return DS
# def constructHNet2(train_met_dis_matrix, met_m):
#     # 计算矩阵的乘积
#     result = np.dot(train_met_dis_matrix, train_met_dis_matrix.T)
#     processed_result = np.where(result > 0, 1, 0)
#     result = np.multiply(processed_result, met_m)
#     return result

# def constructHNet3(train_met_dis_matrix, met_m, dis_m):
#     # 计算矩阵的乘积
#     result1 = np.dot(train_met_dis_matrix, dis_m)
#     result2 = np.dot(result1, train_met_dis_matrix.T)
#     processed_result1 = np.where(result2 > 0, 1, 0)
#     result3 = np.dot(train_met_dis_matrix, train_met_dis_matrix.T)
#     processed_result2 = np.where(result3 > 0, 1, 0)
#     result = processed_result1 + processed_result2
#     result = np.where(result > 0, 1, 0)
#     result = np.multiply(result, met_m)
#     return result