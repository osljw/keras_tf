import tensorflow as tf
from deepctr.input_embedding import preprocess_input_embedding
from deepctr.layers.core import MLP, PredictionLayer, NumericFeatureColumnLayer, EmbeddingFeatureColumnLayer
from deepctr.layers.interaction import CIN
from deepctr.layers.utils import concat_fun
from deepctr.utils import check_feature_config_dict
from tensorflow.python.keras.layers import (Concatenate, Dense, Embedding,
                                            Input, Lambda, Reshape, add)
import tensorflow.keras as K

class FeatureInfo(object):
    def __init__(self, name, dtype, size):
        self.name = name
        self.dtype = dtype
        self.size = size


class EmbeddingFeatureInfo(object):
    def __init__(self, name, num_buckets, dimension):
        self.name = name
        self.num_buckets = num_buckets
        self.dimension = dimension

class NumericFeatureInfo(object):
    def __init__(self, name, size):
        self.name = name
        self.size = size

def build_input(feature_list):
    for feature_info in feature_list:
        name = feature_info.feature_name
        size = feature_info.feature_size
        tensor = tf.keras.layers.Input((size,), name=name)
        feature_info.set_input(tensor)

def build_inputs():
    input_tensors = {}
    input_tensors['uid'] = tf.keras.layers.Input((1,), dtype='int32', name='uid')
    input_tensors['item_id'] = tf.keras.layers.Input((1,), dtype='int32', name='item_id')
    input_tensors['duration_time'] = tf.keras.layers.Input((1,), dtype='float32', name='duration_time')
    return input_tensors

def build_inputs_from_config(feature_list):
    input_tensors = {}
    for feature_info in feature_list:
        name = feature_info.name
        size = feature_info.size
        dtype = feature_info.dtype
        tensor = tf.keras.layers.Input((size,),dtype=dtype, name=name)
        input_tensors[name] = tensor

    print("======== input config =======")
    for name, tensor in input_tensors.items():
        print("\tname: {}, tensor: {}".format(name, tensor))
    return input_tensors

def build_features(input_tensors):
    features = {}
    for feature_name, tensor in input_tensors.items():
        if feature_name in ['duration_time']:
            tensor = tf.string_to_number(tensor, tf.float32)
            features[feature_name] = tensor
        else:
            tensor = tf.reshape(input_tensors[feature_name], [-1])
            tensor =  tf.string_split(tensor, ',')
            sp_values = tf.string_to_number(tensor.values, tf.int32)
            features[feature_name] = tf.SparseTensor(tensor.indices, sp_values, tensor.dense_shape)
    return features

def build_feature_columns(embedding_feature_list):
    fc_dict = {}
    for feature_info in embedding_feature_list:
        name = feature_info.name
        num_buckets = feature_info.num_buckets
        if len(name.split('_X_')) > 1:
            fc_dict[name] = tf.feature_column.crossed_column(name.split('_X_'), hash_bucket_size=num_buckets)
        else:
            fc_dict[name] = tf.feature_column.categorical_column_with_identity(name, num_buckets, default_value=0)
    return fc_dict

def xDeepFM_MTL(input_feature_list, 
                embedding_feature_list,
                numeric_feature_list,
                embedding_size=8, 
                hidden_size=(256, 256), 
                cin_layer_size=(256, 256,),
                cin_split_half=True,
                task_net_size=(128,), 
                l2_reg_linear=0.00001, 
                l2_reg_embedding=0.00001,
                seed=1024, ):
    #video_input = tf.keras.layers.Input((1,), name="video_vec")
    #print("video_input", video_input)
    #video_tensor = FeatureColumnLayer("video_vec")({"video_vec":video_input})
    #print("video_tensor", video_tensor)
    
    input_tensors = build_inputs_from_config(input_feature_list)
    #video_input = input_tensors['video_vec']
    #print("video input:", video_input)
    ##print("embed input:", type(embedding_tensors['video_vec']), embedding_tensors['video_vec'], embedding_tensors['video_vec']._keras_history)

    embedding_size = 8
    features = build_features(input_tensors)
    fc_dict = build_feature_columns(embedding_feature_list)


    with tf.device('/cpu:0'):
        embedding_tensors = {}
        for feature_info in embedding_feature_list:
            name = feature_info.name
            dimension = embedding_size
            inputs = [input_tensors[x] for x in name.split("_X_")]
            tensor = EmbeddingFeatureColumnLayer(name, fc_dict[name], dimension=dimension, features=features)(inputs)
            embedding_tensors[name] = tensor
        
        linear_embedding_tensors = {}
        for feature_info in embedding_feature_list:
            name = feature_info.name
            dimension = 1
            inputs = [input_tensors[x] for x in name.split("_X_")]
            tensor = EmbeddingFeatureColumnLayer(name, fc_dict[name], dimension=dimension, features=features)(inputs)
            linear_embedding_tensors[name] = tensor

    numeric_tensors = {}
    for feature_info in numeric_feature_list:
        name = feature_info.name
        size = feature_info.size
        tensor = NumericFeatureColumnLayer(name, size, features=features)(input_tensors[name])
        #tensor = Dense(embedding_size, use_bias=False, kernel_regularizer=l2(l2_reg), )
        tensor = Dense(embedding_size, use_bias=False)(tensor)
        numeric_tensors[name] = tensor

    linear_input_list = list(linear_embedding_tensors.values())
    linear_input = concat_fun(linear_input_list, axis=1)
    linear_logit = Dense(1)(linear_input)

    deep_emb_list = list(embedding_tensors.values()) + list(numeric_tensors.values())
    deep_emb_list = [Reshape([1, embedding_size])(x) for x in deep_emb_list]

    fm_input = concat_fun(deep_emb_list, axis=1)
    print("=====fm input:", fm_input)

    if len(cin_layer_size) > 0:
        exFM_out = CIN(cin_layer_size, 'relu',
                       cin_split_half, seed)(fm_input)
        exFM_logit = tf.keras.layers.Dense(1, activation=None, )(exFM_out)

    deep_input = tf.keras.layers.Flatten()(fm_input)
    deep_out = MLP(hidden_size)(deep_input)
    print("=====deep input:", deep_input)

    finish_out = MLP(task_net_size)(deep_out)
    finish_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(finish_out)

    like_out = MLP(task_net_size)(deep_out)
    like_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(like_out)

    finish_logit_list = [linear_logit, finish_logit, exFM_logit]
    like_logit_list = [linear_logit, like_logit, exFM_logit]
    #finish_logit_list = [finish_logit, exFM_logit]
    #like_logit_list = [like_logit, exFM_logit]
    finish_logit = tf.keras.layers.add(finish_logit_list)
    like_logit = tf.keras.layers.add(like_logit_list)

    output_finish = PredictionLayer('sigmoid', name='finish')(finish_logit)
    output_like = PredictionLayer('sigmoid', name='like')(like_logit)

    #output_finish = Lambda(lambda x:tf.reduce_sum(x, axis=1, keepdims=True), name="finish")(finish_logit) 
    #output_like = Lambda(lambda x:tf.reduce_sum(x, axis=1, keepdims=True), name="like")(like_logit) 
    #output_finish = Lambda(lambda x:tf.reduce_sum(x, axis=1, keepdims=True, name="finish"))(video_input) 
    #output_like = Lambda(lambda x:tf.reduce_sum(x, axis=1, keepdims=True, name="like"))(video_input) 
    print("output_finish", type(output_finish), output_finish)
    print("output_like", type(output_like), output_like)

    inputs_list = list(input_tensors.values())
    outputs_list = [output_finish, output_like]
    #outputs_list = output_finish
    model = tf.keras.models.Model(inputs=inputs_list, outputs=outputs_list)
    return model

