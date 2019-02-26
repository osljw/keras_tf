import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn import metrics
import tensorflow as tf

from deepctr import SingleFeat
from model import xDeepFM_MTL

ONLINE_FLAG = False
loss_weights = [1, 1, ]  # [0.7,0.3]任务权重可以调下试试
VALIDATION_FRAC = 0.2  # 用做线下验证数据比例

def cal_auc(true, pred):
    auc = metrics.roc_auc_score(true, pred)
    return auc

#column_names = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'did', 'creat_time', 'video_duration']
column_names = ["instance_id", 'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'did', 'creat_time', 'video_duration',
        "words", "freqs",
        "gender", "beauty",
        "pos0", "pos1", "pos2", "pos3"]

if sys.argv[1] == 'track1':
    train_file = '../track1/final_track1_train.txt'
    test_file = '../track1/final_track1_test_no_anwser.txt'
    sparse_features = ['uid', 'item_id', 'author_id', 'item_city', 'channel', 'music_id', 'did', ]
    dense_features = ['video_duration'] 
elif sys.argv[1] == 'track2':
    train_file = '../track2/final_track2_train_new.txt'
    test_file = '../track2/final_track2_test_new.txt'
    offline_train_file = '../track2/xa'
    offline_eval_file = '../track2/xb'
    sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'music_id', 'did', 
                       'words', 'freqs',
                       'gender', 'beauty', 'pos0', 'pos1', 'pos2', 'pos3']
    dense_features = ['video_duration', ] 
elif sys.argv[1] == 'test':
    train_file = 'input/train.txt'
    test_file = 'input/test.txt'
    offline_train_file = 'input/train.txt'
    offline_eval_file = 'input/test.txt'
    #sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'music_id', 'did', ]
    sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'music_id', 'did', 
                       'words', 'freqs',
                       'gender', 'beauty', 'pos0', 'pos1', 'pos2', 'pos3' ]
    dense_features = ['video_duration', ] 

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    tf.keras.backend.get_session().run(tf.local_variables_initializer())
    return auc

if __name__ == "__main__":
    if ONLINE_FLAG:
        data = pd.read_csv(train_file, sep='\t', names=column_names)
        test_data = pd.read_csv(test_file, sep='\t', names=column_names)
        train_size = data.shape[0]
        data = data.append(test_data)
    else:
        #train_size = int(data.shape[0]*(1-VALIDATION_FRAC))
        data = pd.read_csv(offline_train_file, sep='\t', names=column_names)
        eval_data = pd.read_csv(offline_eval_file, sep='\t', names=column_names)
        train_size = data.shape[0]
        data = data.append(eval_data)

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0,)

    target = ['finish', 'like']

    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    print("[main] dense_features:", data[dense_features])

    sparse_feature_list = [SingleFeat(feat, data[feat].nunique())
                           for feat in sparse_features]
    dense_feature_list = [SingleFeat(feat, 0)
                          for feat in dense_features]

    train = data.iloc[:train_size]
    test = data.iloc[train_size:]

    train_model_input = [train[feat.name].values for feat in sparse_feature_list] + \
        [train[feat.name].values for feat in dense_feature_list]
    test_model_input = [test[feat.name].values for feat in sparse_feature_list] + \
        [test[feat.name].values for feat in dense_feature_list]

    train_labels = [train[target[0]].values, train[target[1]].values]
    test_labels = [test[target[0]].values, test[target[1]].values]

    model = xDeepFM_MTL({"sparse": sparse_feature_list,
                         "dense": dense_feature_list})
    model.compile("adagrad", "binary_crossentropy", loss_weights=loss_weights,
                  metrics=[auc])

    #sys.exit()

    if ONLINE_FLAG:
        history = model.fit(train_model_input, train_labels,
                            batch_size=4096, epochs=1, verbose=1)
        pred_ans = model.predict(test_model_input, batch_size=2**14)

    else:
        #history = model.fit(train_model_input, train_labels,
        #                    batch_size=4096, epochs=1, verbose=1, 
        #                    validation_data=(test_model_input, test_labels))
        history = model.fit(train_model_input, train_labels,
                            batch_size=4096, epochs=1, verbose=1)
        pred_ans = model.predict(test_model_input, batch_size=2**14)
        finish_auc = cal_auc(test['finish'], pred_ans[0])
        like_auc = cal_auc(test['like'], pred_ans[1])
        print("finish auc: {}, like auc: {}".format(finish_auc, like_auc))

    if ONLINE_FLAG:
        result = test_data[['uid', 'item_id', 'finish', 'like']].copy()
        result.rename(columns={'finish': 'finish_probability',
                               'like': 'like_probability'}, inplace=True)
        result['finish_probability'] = pred_ans[0]
        result['like_probability'] = pred_ans[1]
        result[['uid', 'item_id', 'finish_probability', 'like_probability']].to_csv(
            'result.csv', index=None, float_format='%.6f')
