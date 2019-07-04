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
from tensorflow.python.client import timeline

from data_input import data_generator
from deepctr.layers.core import MLP, PredictionLayer, NumericFeatureColumnLayer, EmbeddingFeatureColumnLayer
from model import xDeepFM_MTL

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
config.allow_soft_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
K.set_session(sess)  # set this TensorFlow session as the default session for Keras

run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

track = sys.argv[1]
assert track in ["track1", "track2", "test"]
if track == 'track1':
    from config.track1_config import *
elif track == 'track2':
    from config.track2_config import *
elif track == 'test':
    from config.test_config import *

loss_weights = [1, 1, ]  # [0.7,0.3]任务权重可以调下试试

def cal_auc(true, pred):
    auc = metrics.roc_auc_score(true, pred)
    return auc


if __name__ == "__main__":

    model = xDeepFM_MTL(input_feature_list,
                        embedding_feature_list,
                        numeric_feature_list,
                        embedding_size=embedding_size,
                        )
    #with tf.device('/cpu:0'):
    opt = Adagrad(lr=0.02)
    #model.compile("adagrad", "binary_crossentropy", loss_weights=loss_weights)
    #model.compile("adam", "binary_crossentropy", loss_weights=loss_weights)
    
    #with tf.device('/cpu:0'):
    model.compile(opt, "binary_crossentropy", loss_weights=loss_weights)
    #model.compile(opt, "binary_crossentropy", loss_weights=loss_weights, options=run_options, run_metadata=run_metadata)

    #print("==== before model build ===")
    #with tf.device('/cpu:0'):
    #    origin_model = xDeepFM_MTL(input_feature_list,
    #                               embedding_feature_list,
    #                               numeric_feature_list,
    #                               embedding_size=embedding_size,
    #                               )
    #print("==== after model build ===")



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
        
        with tf.device('/cpu:0'):
            history = model.fit_generator(train_generator,
                    steps_per_epoch=train_steps_per_epoch,
                    epochs=1,
                    verbose=1)

        #trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        #with open('timeline.ctf.json', 'w') as f:
        #    f.write(trace.generate_chrome_trace_format())
        
        if not eval_file: continue
        eval_generator = data_generator(eval_file,
                                        column_names=column_names,
                                        features=features,
                                        targets=targets,
                                        batch_size=batch_size,
                                        epochs=1)

        eval_data = []
        for X, Y in eval_generator:
            pred = model.predict(X)
            batch_eval = pd.DataFrame({'uid': X['uid'],
                                       'item_id': X['item_id'],
                                       'finish': Y['finish'],
                                       'like': Y['like'],
                                       'finish_probability': pred[0].reshape(-1),
                                       'like_probability': pred[1].reshape(-1),
            })
            eval_data.append(batch_eval)

        eval_data = pd.concat(eval_data, axis=0)
        print("len: {}".format(len(eval_data)))
        finish_auc = cal_auc(eval_data['finish'], eval_data['finish_probability'])
        like_auc = cal_auc(eval_data['like'], eval_data['like_probability'])
        print("epoch:{}, finish auc: {}, like auc: {}".format(i, finish_auc, like_auc))
        
        # if test has not label, no auc
        #if ONLINE_FLAG: continue

    if ONLINE_FLAG:
        print("\npredict:")
        #pred_ans = model.predict(test_model_input, batch_size=2**14)
        #pred_ans = model.evaluate_generator(eval_generator, steps=3)
        #pred_ans = model.predict_generator(test_generator, steps=3)
        test_generator = data_generator(test_file,
                                        column_names=column_names,
                                        features=features,
                                        targets=targets,
                                        batch_size=batch_size*5,
                                        epochs=1)

        test = []
        for X, Y in test_generator:
            pred = model.predict(X)
            batch_test = pd.DataFrame({'uid': X['uid'],
                                       'item_id': X['item_id'],
                                       'finish': Y['finish'],
                                       'like': Y['like'],
                                       'finish_probability': pred[0].reshape(-1),
                                       'like_probability': pred[1].reshape(-1),
            })
            test.append(batch_test)

        test = pd.concat(test, axis=0)
        print(test.head())
        print("len: {}".format(len(test)))
        #finish_auc = cal_auc(test['finish'], test['finish_probability'])
        #like_auc = cal_auc(test['like'], test['like_probability'])
        #print("epoch:{}, finish auc: {}, like auc: {}".format(i, finish_auc, like_auc))

        output_file = "result/{}_result.csv".format(track)
        test[['uid', 'item_id', 'finish_probability', 'like_probability']].to_csv(
            output_file, index=None)
            #output_file, index=None, float_format='%.12f')
