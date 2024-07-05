import tensorflow as tf
from utils import *
class GraphConvolution():
    def __init__(self,input_dim,output_dim,adj,name,dropout=0.,act=tf.nn.relu):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim,output_dim,name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def __call__(self,inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = tf.nn.dropout(x,1-self.dropout)
            x = tf.matmul(x,self.vars['weights'])
            x = tf.sparse_tensor_dense_matmul(self.adj,x)
            outputs = self.act(x)
        return outputs

class GraphConvolutionSparse():

    def __init__(self, input_dim, output_dim, adj, features_nonzero, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
            x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
            x = tf.sparse_tensor_dense_matmul(self.adj, x)
            outputs = self.act(x)
        return outputs
    
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.units = units
        self.W = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, inputs):
        query = tf.expand_dims(inputs, axis=1)
        score = self.V(tf.nn.tanh(self.W(query)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * query
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector


class InnerProductDecoder():

    def __init__(self,input_dim,name,num_r,dropout=0.,act=tf.nn.relu):
        self.name = name
        self.vars ={}
        self.issparse = False
        self.dropout = dropout
        self.act = act
        self.num_r = num_r
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim,input_dim,name='weights')

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            inputs = tf.nn.dropout(inputs,1-self.dropout)
            R = inputs[0:self.num_r,:]
            D = inputs[self.num_r:,:]
            R = tf.matmul(R,self.vars['weights'])
            D = tf.transpose(D)
            x = tf.matmul(R,D)
            x = tf.reshape(x,[-1])
            outputs = self.act(x)
        return outputs
