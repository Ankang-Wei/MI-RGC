import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from layer import GraphConvolution, GraphConvolutionSparse,  AttentionLayer, InnerProductDecoder
from utils import *
from satt import *
#adj_nonzero4,adj_nonzero5,
class GCNModel():
    def __init__(self, placeholders, num_features, emb_dim, features_nonzero, adj_nonzero1, adj_nonzero2, adj_nonzero3, adj_nonzero4, adj_nonzero5,  num_r, name, act=tf.nn.relu):
        
        self.name = name
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.emb_dim = emb_dim
        self.features_nonzero = features_nonzero
        self.adj_nonzero1 = adj_nonzero1  # 第一个邻接矩阵的非零元素数量
        self.adj_nonzero2 = adj_nonzero2  # 第二个邻接矩阵的非零元素数量
        self.adj_nonzero3 = adj_nonzero3  # 第三个邻接矩阵的非零元素数量
        self.adj_nonzero4 = adj_nonzero4  # 第四个邻接矩阵的非零元素数量
        self.adj_nonzero5 = adj_nonzero5  # 第五个邻接矩阵的非零元素数量
        self.adj1 = placeholders['adj1']  # 第一个邻接矩阵
        self.adj2 = placeholders['adj2']  # 第二个邻接矩阵
        self.adj3 = placeholders['adj3']  # 第三个邻接矩阵
        self.adj4 = placeholders['adj4']  # 第四个邻接矩阵
        self.adj5 = placeholders['adj5']  # 第五个邻接矩阵
        self.dropout = placeholders['dropout']
        self.adjdp = placeholders['adjdp']
        self.act = act
        self.num_r = num_r
        with tf.compat.v1.variable_scope(self.name):
            self.build() 
    def build(self):
        # 第一个邻接矩阵的 Graph Convolution 层
        self.adj1 = dropout_sparse(self.adj1, 1-self.adjdp, self.adj_nonzero1)
        self.hidden1_1 = GraphConvolutionSparse(
            name='gcn_sparse_layer1',
            input_dim=self.input_dim,
            output_dim=self.emb_dim,
            adj=self.adj1,
            features_nonzero=self.features_nonzero,
            dropout=self.dropout,
            act=self.act)(self.inputs)

        self.hidden1_2 = GraphConvolution(
            name='gcn_dense_layer1',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj1,
            dropout=self.dropout,
            act=self.act)(self.hidden1_1)
        self.hidden1_3 = GraphConvolution(
            name='gcn_dense_layer1_2',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj1,
            dropout=self.dropout,
            act=self.act)(self.hidden1_2)
        self.emb1 = GraphConvolution(
            name='gcn_dense_layer1_3',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj1,
            dropout=self.dropout,
            act=self.act)(self.hidden1_3)
        
        # 第二个邻接矩阵的 Graph Convolution 层
        self.adj2 = dropout_sparse(self.adj2, 1-self.adjdp, self.adj_nonzero2)
        self.hidden2_1 = GraphConvolutionSparse(
            name='gcn_sparse_layer2',
            input_dim=self.input_dim,
            output_dim=self.emb_dim,
            adj=self.adj2,
            features_nonzero=self.features_nonzero,  # 使用新的特征非零元素数量
            dropout=self.dropout,
            act=self.act)(self.inputs)

        self.hidden2_2 = GraphConvolution(
            name='gcn_dense_layer2',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj2,
            dropout=self.dropout,
            act=self.act)(self.hidden2_1)
        self.hidden2_3 = GraphConvolution(
            name='gcn_dense_layer2_2',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj2,
            dropout=self.dropout,
            act=self.act)(self.hidden2_2)
        self.emb2 = GraphConvolution(
            name='gcn_dense_layer2_3',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj2,
            dropout=self.dropout,
            act=self.act)(self.hidden2_3)
        

        #第三个邻接矩阵的 Graph Convolution 层
        self.adj3 = dropout_sparse(self.adj3, 1-self.adjdp, self.adj_nonzero3)
        self.hidden3_1 = GraphConvolutionSparse(
            name='gcn_sparse_layer3',
            input_dim=self.input_dim,
            output_dim=self.emb_dim,
            adj=self.adj3,
            features_nonzero=self.features_nonzero,  # 使用新的特征非零元素数量
            dropout=self.dropout,
            act=self.act)(self.inputs)

        self.hidden3_2 = GraphConvolution(
            name='gcn_dense_layer3',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj3,
            dropout=self.dropout,
            act=self.act)(self.hidden3_1)
        self.hidden3_3 = GraphConvolution(
            name='gcn_dense_layer3_2',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj3,
            dropout=self.dropout,
            act=self.act)(self.hidden3_2)
        self.emb3 = GraphConvolution(
            name='gcn_dense_layer3_3',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj3,
            dropout=self.dropout,
            act=self.act)(self.hidden3_3)
        
        # 第四个邻接矩阵的 Graph Convolution 层
        self.adj4 = dropout_sparse(self.adj4, 1-self.adjdp, self.adj_nonzero4)
        self.hidden4_1 = GraphConvolutionSparse(
            name='gcn_sparse_layer4',
            input_dim=self.input_dim,
            output_dim=self.emb_dim,
            adj=self.adj4,
            features_nonzero=self.features_nonzero,  # 使用新的特征非零元素数量
            dropout=self.dropout,
            act=self.act)(self.inputs)

        self.hidden4_2 = GraphConvolution(
            name='gcn_dense_layer4',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj4,
            dropout=self.dropout,
            act=self.act)(self.hidden4_1)
        self.hidden4_3 = GraphConvolution(
            name='gcn_dense_layer4_2',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj4,
            dropout=self.dropout,
            act=self.act)(self.hidden4_2)
        self.emb4 = GraphConvolution(
            name='gcn_dense_layer4_3',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj4,
            dropout=self.dropout,
            act=self.act)(self.hidden4_3)
        
        # 第五个邻接矩阵的 Graph Convolution 层
        self.adj5 = dropout_sparse(self.adj5, 1-self.adjdp, self.adj_nonzero5)
        self.hidden5_1 = GraphConvolutionSparse(
            name='gcn_sparse_layer5',
            input_dim=self.input_dim,
            output_dim=self.emb_dim,
            adj=self.adj5,
            features_nonzero=self.features_nonzero,  # 使用新的特征非零元素数量
            dropout=self.dropout,
            act=self.act)(self.inputs)

        self.hidden5_2 = GraphConvolution(
            name='gcn_dense_layer5',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj5,
            dropout=self.dropout,
            act=self.act)(self.hidden5_1)
        self.hidden5_3 = GraphConvolution(
            name='gcn_dense_layer5_2',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj5,
            dropout=self.dropout,
            act=self.act)(self.hidden5_2)
        self.emb5 = GraphConvolution(
            name='gcn_dense_layer5_3',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj5,
            dropout=self.dropout,
            act=self.act)(self.hidden5_3)

        self.embed1 = GraphAttentionLayer(
            dropout=self.dropout,
            F_=self.emb_dim)(self.emb1)

        self.embed2 = GraphAttentionLayer(
            dropout=self.dropout,
            F_=self.emb_dim)(self.emb2)
        
        self.embed3 = GraphAttentionLayer(
            dropout=self.dropout,
            F_=self.emb_dim)(self.emb3)

        self.embed4 = GraphAttentionLayer(
            dropout=self.dropout,
            F_=self.emb_dim)(self.emb4)
                               
        self.embed5 = GraphAttentionLayer(
            dropout=self.dropout,
            F_=self.emb_dim)(self.emb5)
        
        self.embed6 = GraphAttentionLayer(
            dropout=self.dropout,
            F_=self.emb_dim)(self.hidden1_1)
        
        self.embed7 = GraphAttentionLayer(
            dropout=self.dropout,
            F_=self.emb_dim)(self.hidden1_2)
       
        self.embeddings = self.embed1+ self.embed2+ self.embed3 + self.embed4 +self.embed5 + self.embed6+ self.embed7
        self.reconstructions = InnerProductDecoder(
            name='gcn_decoder',
            input_dim=self.emb_dim, num_r=self.num_r, act=tf.nn.sigmoid)(self.embeddings)
# 