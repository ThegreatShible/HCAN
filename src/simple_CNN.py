import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
#from tensorflow.keras import Model


def bulid_simple_CNN(embedding_matrix, max_text_length, filters_sizes, n_filters, activation_func, p, wv_trainable, nb_outputs) :
    """Build a simple 1D CNN model for text classification
    
    Arguments:
        embedding_matrix {2D numpy array} -- [words' embedding matrix]
        max_text_length {Integer} -- [Length of the input sequence]
        filters_sizes {Integer} -- [Size of the 1D Conv filters]
        n_filters {Integer} -- [Number of 1D Conv filters]
        activation_func {String} -- [Activation function of the onv layers. One of the Keras activation function names]
        p {Float} -- [Probability of dropout]
        wv_trainable {Boolean} -- [if True then the embedding matrix is trainable]
        nb_outputs {Integer} -- [Number of outputs (number of labels)]
    
    Returns:
        [type] -- [description]
    """
    input_words = Input(shape=(max_text_length,), name="words")
    vocab_length, word_dim = embedding_matrix.shape
    embedding = Embedding(input_dim = vocab_length, output_dim= word_dim, 
                        input_length=max_text_length, weights = [embedding_matrix], 
                        trainable = wv_trainable, name="word_embedding")(input_words)

    filters = []
    for filter_size in filters_sizes : 
        conv = Conv1D(n_filters, kernel_size=filter_size,
                        padding="valid", activation= activation_func, name="convolution_%d" % filter_size)(embedding)
        pooling  = GlobalMaxPooling1D(name="pooling_%d" % filter_size)(conv)
        filters.append(pooling)
    merge_conv = concatenate(filters, name="merge_convolutions")
    drop = Dropout(p, name="drop")(merge_conv)
    dense = Dense(nb_outputs, name="dense_layer")(drop)
    classif = Softmax(name="softmax")(dense)
    model  =  tf.keras.Model(inputs=input_words, outputs = classif)
    return model