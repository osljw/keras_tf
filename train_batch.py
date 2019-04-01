import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn import metrics
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.optimizers import Adagrad

from deepctr import SingleFeat
from deepctr.layers.core import MLP, PredictionLayer, NumericFeatureColumnLayer, EmbeddingFeatureColumnLayer
from model import xDeepFM_MTL, FeatureInfo, EmbeddingFeatureInfo, NumericFeatureInfo

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
config.allow_soft_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
K.set_session(sess)  # set this TensorFlow session as the default session for Keras

if sys.argv[1] == 'track1':
    from track1_config import *
elif sys.argv[1] == 'track2':
    from track2_config import *
elif sys.argv[1] == 'test':
    from test_config import *


loss_weights = [1, 1, ]  # [0.7,0.3]任务权重可以调下试试
VALIDATION_FRAC = 0.2  # 用做线下验证数据比例

def cal_auc(true, pred):
    auc = metrics.roc_auc_score(true, pred)
    return auc

def duration_min_max(x):
    return (x-0)/(duration_time_max - 0)

def data_clip(x):
    return x if x > 0 else 1

def data_preprocess(df):
    #print("sparse_features", sparse_features)
    df['duration_time'] = df['duration_time'].apply(duration_min_max)
    #df[sparse_features] = df[sparse_features].apply(lambda x: x.clip(lower=0))
    #df[dense_features] = df[dense_features].fillna(0,)
    df = df.fillna(0)
    return df

#def data_generator(file_name):
#    fd = open(file_name)
#    reader = pd.read_csv(fd, sep='\t', chunksize=batch_size, names=column_names, header=None)
#    while True:
#        for chunk_df in reader:
#            #print("dtypes:", chunk_df.dtypes)
#            #print("data before modify:", chunk_df['user_city'].head())
#            chunk_df = data_preprocess(chunk_df)
#            #print("data after modify:", df['user_city'].head())
#            X = [chunk_df[feat.name].values for feat in sparse_feature_list] + \
#                    [chunk_df[feat.name].values for feat in dense_feature_list]
#            Y = [chunk_df[target[0]].values, chunk_df[target[1]].values]
#            yield X, Y
#        fd.close()
#        fd = open(file_name)
#        reader = pd.read_csv(fd, sep='\t', chunksize=batch_size, names=column_names, header=None)

features = [
    "uid",
    "item_id",
    "author_id",
    "item_city", 
    "channel",
    "music_id",
    "words",
    "duration_time",
    ]
embedding_size = 8

print_head = True
def data_generator(file_name):
    global print_head
    fd = open(file_name)
    reader = pd.read_csv(fd, sep='\t', chunksize=batch_size, names=column_names, header=None)
    while True:
        for chunk_df in reader:
            #print("dtypes:", chunk_df.dtypes)
            #print("data before modify:", chunk_df['user_city'].head())
            chunk_df = data_preprocess(chunk_df)
            #print("data after modify:", df['user_city'].head())
            #X = [chunk_df[feat.name].values for feat in sparse_feature_list] + \
            #        [chunk_df[feat.name].values for feat in dense_feature_list]
            X = {feature: chunk_df[feature].astype(str).values for feature in features}
            Y = {"finish":chunk_df[target[0]].values, "like":chunk_df[target[1]].values}
            if print_head == True:
                print("columns:", list(chunk_df.columns))
                print(chunk_df.head())
                print("X", X)
                print_head = False
            yield X, Y
        fd.close()
        fd = open(file_name)
        reader = pd.read_csv(fd, sep='\t', chunksize=batch_size, names=column_names, header=None)


if __name__ == "__main__":

    #test = pd.read_csv(test_file, sep='\t', names=column_names)
    #test = data_preprocess(test)
    #test_model_input = [test[feat.name].values for feat in sparse_feature_list] + \
    #    [test[feat.name].values for feat in dense_feature_list]

    train_generator = data_generator(train_file)
    test_generator = data_generator(test_file)

    input_feature_list = [FeatureInfo(feature, "string", 1) for feature in features]
    
    embedding_feature_list = [
        EmbeddingFeatureInfo('uid', 100000, dimension=embedding_size),
        EmbeddingFeatureInfo('item_id', 100000, dimension=embedding_size),
        EmbeddingFeatureInfo('author_id', 100000, dimension=embedding_size),
        EmbeddingFeatureInfo('item_city', 100000, dimension=embedding_size),
        EmbeddingFeatureInfo('channel', 100000, dimension=embedding_size),
        EmbeddingFeatureInfo('music_id', 100000, dimension=embedding_size),
        EmbeddingFeatureInfo('words', 100000, dimension=embedding_size),
        #EmbeddingFeatureInfo('_X_'.join(['uid', 'item_id']), 100000, dimension=embedding_size),
    ]
    
    numeric_feature_list = [
        NumericFeatureInfo('duration_time', 1),
    ]

    print("==== before model build ===")
    with tf.device('/cpu:0'):
        origin_model = xDeepFM_MTL(input_feature_list,
                                   embedding_feature_list,
                                   numeric_feature_list,
                                   embedding_size=embedding_size,
                                   )
    print("==== after model build ===")
    #opt = Adagrad(lr=0.08)

    K.get_session().run(tf.global_variables_initializer())
    K.get_session().run(tf.tables_initializer())

    for layer in origin_model.layers:
        print("\tname:", layer.name, 
              "trainable_weights:", layer.trainable_weights,
              "weights:", layer.weights,
              )
              #"weights:", layer.get_weights())

    if 1:
        model = origin_model
        model.compile("adagrad", "binary_crossentropy", loss_weights=loss_weights)
    else:
        #model = multi_gpu_model(origin_model, gpus=3, cpu_relocation=True)
        model = multi_gpu_model(origin_model, gpus=ngpus)
        model.compile("adagrad", "binary_crossentropy", loss_weights=loss_weights)
        #model.compile("adam", "binary_crossentropy", loss_weights=loss_weights)
        #model.compile(opt, "binary_crossentropy", loss_weights=loss_weights)
    
    print("ngpus: {}".format(ngpus))
    print("epochs: {}".format(epochs))
    print("batch_size: {}".format(batch_size))
    print("train_steps_per_epoch: {}".format(train_steps_per_epoch))

    for i in range(epochs):
        print("\n\n========== epochs: {} ============".format(i))
        print("train:")
        history = model.fit_generator(train_generator,
                steps_per_epoch=train_steps_per_epoch,
                epochs=1,
                verbose=2)
        print("\npredict:")
        #pred_ans = model.predict(test_model_input, batch_size=2**14)
        #pred_ans = model.evaluate_generator(eval_generator, steps=3)
        pred_ans = model.predict_generator(test_generator, steps=3)
        #if ONLINE_FLAG: continue
        #if ONLINE_FLAG:
        #    result = test[['uid', 'item_id', 'finish', 'like']].copy()
        #    result.rename(columns={'finish': 'finish_probability',
        #                           'like': 'like_probability'}, inplace=True)
        #    result['finish_probability'] = pred_ans[0]
        #    result['like_probability'] = pred_ans[1]
        #    output_file = 'epoch' + str(i) + '_result.csv'
        #    result[['uid', 'item_id', 'finish_probability', 'like_probability']].to_csv(
        #        output_file, index=None, float_format='%.6f')
        #else:
        #    finish_auc = cal_auc(test['finish'], pred_ans[0])
        #    like_auc = cal_auc(test['like'], pred_ans[1])
        #    print("epoch:{}, finish auc: {}, like auc: {}".format(i, finish_auc, like_auc))

        #finish_auc = cal_auc(test['finish'], pred_ans[0])
        #like_auc = cal_auc(test['like'], pred_ans[1])
        #print("epoch:{}, finish auc: {}, like auc: {}".format(i, finish_auc, like_auc))

    #sys.exit()

    #if ONLINE_FLAG:
    #    result = test[['uid', 'item_id', 'finish', 'like']].copy()
    #    result.rename(columns={'finish': 'finish_probability',
    #                           'like': 'like_probability'}, inplace=True)
    #    result['finish_probability'] = pred_ans[0]
    #    result['like_probability'] = pred_ans[1]
    #    result[['uid', 'item_id', 'finish_probability', 'like_probability']].to_csv(
    #        'result.csv', index=None, float_format='%.6f')
