import numpy as np
import scipy.sparse as sp
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.framework import  ops
ops.reset_default_graph()
import gc
import random
from layer import *
from metric import cv_model_evaluate
from utils import *
from model import GCNModel
from opt import Optimizer

def PredictScore(train_met_dis_matrix, met_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr, adjdp, simwh,mat_m):
    np.random.seed(seed)
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(seed)

    #train_met_dis_matrix = constructHNetPHP(train_met_dis_matrix, 0.99*mat_m)
    dis_mat = np.where(dis_matrix < simwh, 0, dis_sim)
    
    # 构造第一个邻接矩阵 adj1
    met_mat1 = np.where(met_matrix < 0.995, 0, met_sim)
    adj1 = constructHNet(train_met_dis_matrix, met_mat1, dis_mat)
    adj1 = sp.csc_matrix(adj1)
    
    #构造第二个邻接矩阵 adj2
    met_mat2 = np.where((met_matrix > 0.995) | (met_matrix < 0.99), 0, met_matrix)
    adj2 = constructHNet(train_met_dis_matrix, met_mat2, dis_mat)  # 假设有一个新的构造邻接矩阵的函数
    adj2 = sp.csc_matrix(adj2)

    #构造第一个邻接矩阵 adj3
    met_mat3 = np.where((met_matrix > 0.99) | (met_matrix < 0.985), 0, met_matrix)
    adj3 = constructHNet(train_met_dis_matrix, met_mat3, dis_mat)
    adj3 = sp.csc_matrix(adj3)
    
    ##构造第二个邻接矩阵 adj4
    met_mat4 = np.where((met_matrix > 0.985) | (met_matrix < 0.98), 0, met_matrix)
    adj4 = constructHNet(train_met_dis_matrix, met_mat4, dis_mat)  # 假设有一个新的构造邻接矩阵的函数
    adj4 = sp.csc_matrix(adj4)

    # # 构造第二个邻接矩阵 adj5
    met_mat5 = np.where((met_matrix > 0.98) | (met_matrix < 0.97), 0, met_matrix)
    adj5 = constructHNet(train_met_dis_matrix, met_mat5, dis_mat)  # 假设有一个新的构造邻接矩阵的函数
    adj5 = sp.csc_matrix(adj5)


    #构造第二个邻接矩阵 adj5
    met_mat5 = np.multiply(mat_m,met_matrix)
    adj5 = constructHNet(train_met_dis_matrix, met_mat5, dis_mat)  # 假设有一个新的构造邻接矩阵的函数
    adj5 = sp.csc_matrix(adj5)
    
    
    association_nam = train_met_dis_matrix.sum()
    X = constructNet(train_met_dis_matrix)
    features = sparse_to_tuple(sp.csc_matrix(X))
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    adj_orig = train_met_dis_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csc_matrix(adj_orig))

    adj_norm1 = preprocess_graph(adj1)
    adj_norm2 = preprocess_graph(adj2)
    adj_norm3 = preprocess_graph(adj3)
    adj_norm4 = preprocess_graph(adj4)
    adj_norm5 = preprocess_graph(adj5)
    adj_nonzero1 = adj_norm1[1].shape[0]
    adj_nonzero2 = adj_norm2[1].shape[0]
    adj_nonzero3 = adj_norm3[1].shape[0]
    adj_nonzero4 = adj_norm4[1].shape[0]
    adj_nonzero5 = adj_norm5[1].shape[0]

    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj1': tf.sparse_placeholder(tf.float32),
        'adj2': tf.sparse_placeholder(tf.float32),
        'adj3': tf.sparse_placeholder(tf.float32),
        'adj4': tf.sparse_placeholder(tf.float32),
        'adj5': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'adjdp': tf.placeholder_with_default(0., shape=())
    }

    
    # 使用两个不同的邻接矩阵 adj1 和 adj2 来构建 GCN 模型
    model = GCNModel(placeholders, num_features, emb_dim,
                     features_nonzero, adj_nonzero1, adj_nonzero2, adj_nonzero3, adj_nonzero4, adj_nonzero5,train_met_dis_matrix.shape[0], name='GCNGAT')
    #,adj_nonzero4,adj_nonzero5
    with tf.name_scope('optimizer'):
        print("Logits Shape:", model.embeddings.shape)
        print("Labels Shape:", tf.reshape(tf.sparse_tensor_to_dense(
            placeholders['adj_orig'], validate_indices=False), [-1]).shape)
    
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(
                placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model,
            lr=lr,num_u=train_met_dis_matrix.shape[0],num_v=train_met_dis_matrix.shape[1],association_nam=association_nam)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['adj1']: adj_norm1})
        feed_dict.update({placeholders['adj2']: adj_norm2})
        feed_dict.update({placeholders['adj3']: adj_norm3})
        feed_dict.update({placeholders['adj4']: adj_norm4})
        feed_dict.update({placeholders['adj5']: adj_norm5})
        feed_dict.update({placeholders['adj_orig']: adj_orig})
        feed_dict.update({placeholders['dropout']: dp})
        feed_dict.update({placeholders['adjdp']: adjdp})
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
        if epoch % 100 == 0:
            feed_dict.update({placeholders['dropout']: 0})
            feed_dict.update({placeholders['adjdp']: 0})
            res = sess.run(model.reconstructions, feed_dict=feed_dict)
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(avg_cost))
    print('Optimization Finished!')
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjdp']: 0})
    res = sess.run(model.reconstructions, feed_dict=feed_dict)
    sess.close()
    return res


def cross_validation_experiment(met_dis_matrix, met_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr, adjdp,simwh,mat_m):
    #进行交叉验证
    index_matrix = np.mat(np.where(met_dis_matrix == 1))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)
    random.shuffle(random_index)
    k_folds = 5
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                 k_folds]).reshape(k_folds, CV_size,  -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
        random_index[association_nam - association_nam % k_folds:]
    random_index = temp
    metric = np.zeros((1, 7))
    print("seed=%d, evaluating met-disease...." % (seed))
    for k in range(k_folds):
        print("------this is %dth cross validation------" % (k+1))
        train_matrix = np.matrix(met_dis_matrix, copy=True)
        train_matrix[tuple(np.array(random_index[k]).T)] = 0
        met_len = met_dis_matrix.shape[0]
        dis_len = met_dis_matrix.shape[1]
        met_disease_res = PredictScore(
            train_matrix, met_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr,  adjdp,simwh,mat_m)
        predict_y_proba = met_disease_res.reshape(met_len, dis_len)

        metric_tmp = cv_model_evaluate(
            met_dis_matrix, predict_y_proba, train_matrix)
        print(metric_tmp)
        metric += metric_tmp
        del train_matrix
        gc.collect()
    print(metric / k_folds)
    metric = np.array(metric / k_folds)
    return metric

if __name__ == "__main__":
    met_sim = np.loadtxt('data/V.csv', delimiter=',')
    dis_sim = np.loadtxt('data/H.csv', delimiter=',')
    met_dis_matrix = np.loadtxt('data/VH.csv', delimiter=',')
    mat_m = np.loadtxt('data/high2-adj.csv', delimiter=',')
    epoch = 4000
    emb_dim = 64
    lr = 0.01
    adjdp = 0.5
    dp = 0.3
    simwh = 1
    result = np.zeros((1, 7), float)
    average_result = np.zeros((1, 7), float)
    circle_time = 1
    for i in range(circle_time):
        result += cross_validation_experiment(
            met_dis_matrix, met_sim, dis_sim, i, epoch, emb_dim, dp, lr, adjdp,simwh,mat_m)
    average_result = result / circle_time
    print(average_result)

