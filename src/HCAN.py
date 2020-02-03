import tensorflow as tf
from tensorflow import keras
import  tensorflow.keras.backend as K
from tensorflow.keras.layers import *
import numpy as np
import math
from tensorflow.keras import Model

class SelfAttention(Layer) :
    
    def __init__(self ,h, **kwargs) : 
        """ Constructor for SelfAttention
        Arguments:
            h {Integer} -- The number of splits to do multihead attention
        """
        super(SelfAttention, self).__init__(**kwargs)
        self.h = h

    def call(self, input) : 
        """Performs the self attention
        
        Arguments:
            input {List of matrix} -- The input is the list containing the Q,K and V matrix respectively
        """
        Q = input[0]
        K = input[1]
        V = input[2]
        d = Q.get_shape().as_list()[-1]
        sq = tf.split(Q,self.h, -1)
        sk = tf.split(K,self.h, -1)
        sv = tf.split(V,self.h, -1)

        splitted_res = []
        for i in range(0,self.h) : 
            sqi = sq[i]
            ski = sk[i]
            svi = sv[i]
            qk = tf.matmul(sqi, ski, transpose_b=True)
            normed_qk = tf.math.divide(qk, math.sqrt(d))
            #TODO : Axis of softmax
            soft = tf.nn.softmax(normed_qk, axis=-1)
            split_res = tf.matmul(soft, svi)
            splitted_res.append(split_res)
        res = tf.concat(splitted_res, -1)
        return res

    def _split(self, input,h) :
        t_input, dim = (input,-1)
        ex_input = K.expand_dims(t_input, 0)
        split = tf.split(ex_input, h, dim)
        res = tf.concat(split, 0)
        return res

    def _group(self, input) : 
        dim0= input.get_shape().as_list()[0]
        split = tf.split(input, dim0, 0)
        grouped = tf.concat(split, -1)
        squeezed = tf.squeeze(grouped, axis= 0)
        return squeezed



class TargetAttention(SelfAttention) : 

    def __init__(self ,h, **kwargs) : 
        """ Constructor for TargetAttention
        Arguments:
            h {Integer} -- The number of splits to do multihead attention
        """
        super(TargetAttention, self).__init__(h,**kwargs)

    def build(self, input_shape):
        inshape = input_shape[-1][-1]
        self.T = self.add_weight(shape=(1,1,inshape),
                                initializer='random_normal',
                                trainable=True)

    def call(self,input) : 
        """Performs the target attention
        
        Arguments:
            input {List of matrix} -- The input is the list containing the K and V matrix respectively
        """
        K = input[0]
        V = input[1]
        
        d = K.get_shape().as_list()[-1]
        st = self._split(self.T,self.h)
        sk = self._split(K,self.h)
        sv = self._split(V,self.h)
        qk = tf.matmul(st, sk, transpose_b=True)
        normed_qk = tf.math.divide(qk, math.sqrt(d))
        #TODO : Axis of softmax
        #Alpha dans ELU
        soft = tf.nn.softmax(normed_qk, axis=-1)
        drop_soft = tf.nn.dropout(x = soft, rate = 0.1)
        split_res = tf.matmul(soft, sv)
        res = self._group(split_res)
        return res


class HierarchyLayer(Layer) : 
    def __init__(self, l, d, number_of_splits, hierarchy_number=1, **kwargs):
        """HierarchyLayer represents all the layers from position embedding to target attention. 
        The output of that layer could be inserted into the input of another HierarchyLayer to have a two hierarchical system.
        
        Arguments:
            
            l {Integer} -- [length of the input sequence]
            d {Integer} -- [dimension of the input embeddings]
            number_of_splits {Integer} -- [number of splits in the multihead attention layers. Must be a divisor of d]
            hierarchy_number {Integer} -- 
        
        Keyword Arguments:
            hierarchy_number {Integer} -- [Hierarchy number to differentiate the hierarchies in a multi-hierarchy system] (default: {1})
        """
        super(HierarchyLayer, self).__init__(**kwargs)
        self.positions  = tf.keras.backend.constant(np.arange(l),dtype=np.int32)
        self.position_embedding = tf.keras.layers.Embedding(input_dim=l, output_dim=d, 
                                    input_shape=(l,), trainable=True, name="position_embedding_%d" % hierarchy_number )(self.positions)
        self.position_embedding = tf.expand_dims(self.position_embedding,0)


        self.fused_embeddings = tf.keras.layers.Add(name="embeddings_with_position_%d" % hierarchy_number)
        self.drop_embeddings = tf.keras.layers.Dropout(0.1, name="fused_embeddings_dropout_%d" % hierarchy_number)
        self.Qa = Conv1D(d, kernel_size=3,
                        padding="same", activation= tf.keras.activations.elu, name="Qa_%d" % hierarchy_number)
        self.Qb = Conv1D(d, kernel_size=3,
                        padding="same", activation= tf.keras.activations.elu, name="Qb_%d" % hierarchy_number)
        self.Ka = Conv1D(d, kernel_size=3,
                        padding="same", activation= tf.keras.activations.elu, name="Ka_%d" % hierarchy_number)
        self.Kb = Conv1D(d, kernel_size=3,
                        padding="same", activation= tf.keras.activations.elu, name="Kb_%d" % hierarchy_number)
        self.Va = Conv1D(d, kernel_size=3,
                        padding="same", activation= tf.keras.activations.elu, name="Va_%d" % hierarchy_number)
        self.Vb = Conv1D(d, kernel_size=3,
                        padding="same", activation= tf.keras.activations.tanh, name="Vb_%d" % hierarchy_number)

        self.self_attention_elu = SelfAttention(h = number_of_splits, name="self_attention_elu_%d" % hierarchy_number )
        self.self_attention_tanh =SelfAttention(h = number_of_splits, name="self_attention_tanh_%d" % hierarchy_number)
        
        self.fused_attention = tf.keras.layers.Multiply(name="fused_attention_%d" % hierarchy_number)
        self.normalised_attention = tf.keras.layers.LayerNormalization(name="layer_normalization_%d" % hierarchy_number)

        self.K = Conv1D(d, kernel_size=3,
                        padding="same", activation= tf.keras.activations.elu, name="K_%d" % hierarchy_number)
        self.V = Conv1D(d, kernel_size=3,
                        padding="same", activation= tf.keras.activations.elu, name="V_%d" % hierarchy_number)

        self.target_attention = TargetAttention(h=number_of_splits, name="target_attention_%d" % hierarchy_number)


    def build(self, input_shape):
        self.built = True


    def call(self, input):

        fe = self.fused_embeddings ([input, self.position_embedding])
        de = self.drop_embeddings (fe)
        qa = self.Qa =(de)
        qb = self.Qb =(de)
        ka =self.Ka(de)
        kb = self.Kb(de)
        va = self.Va(de)
        vb =self.Vb(de)
        sae = self.self_attention_elu([qa, ka, va])
        sat = self.self_attention_tanh([qb, kb, vb])
        fa =self.fused_attention([sae, sat])
        na = self.normalised_attention(fa)
        k = self.K(na)
        v = self.V(na)
        result = self.target_attention ([k,v])
        return result  




def build_HCAN(input_dim, word_embedding_matrix, number_of_classes,number_of_splits,em_mat_trainable=True) : 
    """Builds an HCAN model with 1 or 2 hierarchies depending on the format of `input_dim`
    
    Arguments:
        input_dim {Integer or tuple,list or numpy array} -- [Length of the input sequence if there is one hierarchy. Tuple, list or numpy array with 2 elements representing the dimensions if there are 2 hierarchies]
        word_embedding_matrix {2D numpy array} -- [the words embedding matrix]
        number_of_classes {Integer} -- [Number of possible labels (for classification)]
        number_of_splits {Integer} -- [Number of splits for the multihead attention. Must be a divisor of the word_embedding_matrix second dimension (words' embedding dimension)]

    
    Keyword Arguments:
        em_mat_trainable {bool} -- [True if the embedding matrix is trainable] (default: {True})
    """
    vocab_length, embedding_dim = word_embedding_matrix.shape
    if embedding_dim % number_of_splits != 0 : 
        raise Exception("Number of splits {} must be a divisor of the embedding dimension {}".format( number_of_splits, embedding_dim))
    
    if type(input_dim) == int : 
        l = input_dim
        input_words = Input(shape=(l,), name="input")
        word_embedding = Embedding(input_dim = vocab_length, output_dim= embedding_dim, 
                            input_length=l, weights = [word_embedding_matrix], 
                            trainable = em_mat_trainable, name="word_embedding")(input_words)
        hierarchy = HierarchyLayer( l, embedding_dim, number_of_splits, hierarchy_number=1, name ="simple_hierarchy")
        res_hierarchy = hierarchy(word_embedding)
        res_hierarchy =  tf.squeeze(res_hierarchy, axis=-2)

  
    elif type(input_dim)  ==  list or type(input_dim) == np.ndarray or type(input_dim) == tuple : 

        l2, l1 = (input_dim[0], input_dim[1])
        input_words = Input(shape=(l2,l1), name="input")
        word_embedding = Embedding(input_dim = vocab_length, output_dim= embedding_dim, 
                                input_length=l1, weights = [word_embedding_matrix], 
                                trainable = em_mat_trainable, name="word_embedding")(input_words)

        h1 =  HierarchyLayer( l1, embedding_dim, number_of_splits, hierarchy_number=1, name ="fist_hierarchy")
        h2 =  HierarchyLayer( l2, embedding_dim, number_of_splits, hierarchy_number=2, name ="second_hierarchy")
        splitted_phrases = tf.split(word_embedding, l2, axis = -3 )
        first_res = [h1(tf.squeeze(y, axis=-3)) for y in splitted_phrases]
        grouped_first_res = tf.concat(first_res, axis =-2)
        res_hierarchy = h2(grouped_first_res)
        res_hierarchy = tf.squeeze(res_hierarchy, -2)
    
    dense = tf.keras.layers.Dense(number_of_classes,  name="dense_layer")(res_hierarchy)
    classif = tf.keras.layers.Softmax(name="softmax")(dense)
    return Model(inputs=input_words, outputs = classif)