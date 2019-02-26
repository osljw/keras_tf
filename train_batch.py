import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn import metrics
import tensorflow as tf

from deepctr import SingleFeat
from model import xDeepFM_MTL

loss_weights = [1, 1, ]  # [0.7,0.3]任务权重可以调下试试
VALIDATION_FRAC = 0.2  # 用做线下验证数据比例

def cal_auc(true, pred):
    auc = metrics.roc_auc_score(true, pred)
    return auc

def data_generator(file_name):
    fd = open(file_name)
    reader = pd.read_csv(fd, sep='\t', chunksize=batch_size, names=column_names, header=None)
    while True:
        for chunk_df in reader:
            #print("data:", chunk_df.head())
            chunk_df['duration_time'] = chunk_df['duration_time'].apply(lambda x: (x-0)/(duration_time_max -0))
            X = [chunk_df[feat.name].values for feat in sparse_feature_list] + \
                    [chunk_df[feat.name].values for feat in dense_feature_list]
            Y = [chunk_df[target[0]].values, chunk_df[target[1]].values]
            yield X, Y
        fd.close()
        fd = open(file_name)
        reader = pd.read_csv(fd, sep='\t', chunksize=batch_size, names=column_names, header=None)

if sys.argv[1] == 'track1':
    from track1_config import *
elif sys.argv[1] == 'track2':
    from track2_config import *
elif sys.argv[1] == 'test':
    from test_config import *

if __name__ == "__main__":

    test = pd.read_csv(test_file, sep='\t', names=column_names)
    test['duration_time'] = test['duration_time'].apply(lambda x: (x-0)/(640 -0))
    test_model_input = [test[feat.name].values for feat in sparse_feature_list] + \
        [test[feat.name].values for feat in dense_feature_list]

    train_generator = data_generator(train_file)
    model = xDeepFM_MTL({"sparse": sparse_feature_list,
                         "dense": dense_feature_list})
    model.compile("adagrad", "binary_crossentropy", loss_weights=loss_weights)
    print("batch_size: {}".format(batch_size))
    print("train_steps_per_epoch: {}".format(train_steps_per_epoch))

    epochs = 5
    for i in range(epochs):
        history = model.fit_generator(train_generator,
                steps_per_epoch=train_steps_per_epoch,
                epochs=1,
                verbose=2)
        pred_ans = model.predict(test_model_input, batch_size=2**14)
        #pred_ans = model.predict_generator(test_generator, steps=3)
        finish_auc = cal_auc(test['finish'], pred_ans[0])
        like_auc = cal_auc(test['like'], pred_ans[1])
        print("epoch:{}, finish auc: {}, like auc: {}".format(i, finish_auc, like_auc))

    #sys.exit()

    if ONLINE_FLAG:
        result = test[['uid', 'item_id', 'finish', 'like']].copy()
        result.rename(columns={'finish': 'finish_probability',
                               'like': 'like_probability'}, inplace=True)
        result['finish_probability'] = pred_ans[0]
        result['like_probability'] = pred_ans[1]
        result[['uid', 'item_id', 'finish_probability', 'like_probability']].to_csv(
            'result.csv', index=None, float_format='%.6f')
