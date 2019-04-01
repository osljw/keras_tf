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

from data_input import data_generator
from deepctr.layers.core import MLP, PredictionLayer, NumericFeatureColumnLayer, EmbeddingFeatureColumnLayer
from model import xDeepFM_MTL

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
    from config.test_config import *


loss_weights = [1, 1, ]  # [0.7,0.3]任务权重可以调下试试
VALIDATION_FRAC = 0.2  # 用做线下验证数据比例

def cal_auc(true, pred):
    auc = metrics.roc_auc_score(true, pred)
    return auc



if __name__ == "__main__":


    model = xDeepFM_MTL(input_feature_list,
                        embedding_feature_list,
                        numeric_feature_list,
                        embedding_size=embedding_size,
                        )
    model.compile("adagrad", "binary_crossentropy", loss_weights=loss_weights)

    #print("==== before model build ===")
    #with tf.device('/cpu:0'):
    #    origin_model = xDeepFM_MTL(input_feature_list,
    #                               embedding_feature_list,
    #                               numeric_feature_list,
    #                               embedding_size=embedding_size,
    #                               )
    #print("==== after model build ===")
    ##opt = Adagrad(lr=0.08)



    #if 1:
    #    model = origin_model
    #    model.compile("adagrad", "binary_crossentropy", loss_weights=loss_weights)
    #else:
    #    #model = multi_gpu_model(origin_model, gpus=3, cpu_relocation=True)
    #    model = multi_gpu_model(origin_model, gpus=ngpus)
    #    model.compile("adagrad", "binary_crossentropy", loss_weights=loss_weights)
    #    #model.compile("adam", "binary_crossentropy", loss_weights=loss_weights)
    #    #model.compile(opt, "binary_crossentropy", loss_weights=loss_weights)
    
    print("ngpus: {}".format(ngpus))
    print("epochs: {}".format(epochs))
    print("batch_size: {}".format(batch_size))
    print("train_steps_per_epoch: {}".format(train_steps_per_epoch))
    
    K.get_session().run(tf.global_variables_initializer())
    K.get_session().run(tf.tables_initializer())

    
    for layer in model.layers:
        print("\tname:", layer.name, 
              "trainable_weights:", layer.trainable_weights,
              "weights:", layer.weights,
              )
              #"weights:", layer.get_weights())
    print("model inputs:", model.inputs)
    print("model outputs:", model.outputs)

    for i in range(epochs):
        print("\n\n========== epochs: {} ============".format(i))
        print("train:")
        #train_generator = data_generator(train_file, epochs=1)
        train_generator = data_generator(train_file, 
                                         column_names=column_names,
                                         features=features,
                                         targets=targets,
                                         batch_size=batch_size,
                                         epochs=1)
        history = model.fit_generator(train_generator,
                steps_per_epoch=train_steps_per_epoch,
                epochs=1,
                verbose=2)
        print("\npredict:")
        #pred_ans = model.predict(test_model_input, batch_size=2**14)
        #pred_ans = model.evaluate_generator(eval_generator, steps=3)
        #pred_ans = model.predict_generator(test_generator, steps=3)
        test_generator = data_generator(test_file,
                                        column_names=column_names,
                                        features=features,
                                        targets=targets,
                                        batch_size=batch_size,
                                        epochs=1)
        test = []
        for X, Y in test_generator:
            pred = model.predict(X)
            batch_test = pd.DataFrame({'uid': X['uid'],
                                       'item_id': X['item_id'],
                                       'finish': Y['finish'],
                                       'like': Y['like'],
                                       'finish_pred': pred[0].reshape(-1),
                                       'like_pred': pred[1].reshape(-1),
            })
            test.append(batch_test)
        test = pd.concat(test, axis=0)
        print(test.head())
        print("len: {}".format(len(test)))
        finish_auc = cal_auc(test['finish'], test['finish_pred'])
        like_auc = cal_auc(test['like'], test['like_pred'])
        print("epoch:{}, finish auc: {}, like auc: {}".format(i, finish_auc, like_auc))

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
