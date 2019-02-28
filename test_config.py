import os
from deepctr import SingleFeat

ngpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

column_names = ["instance_id", 'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'device', 'creat_time', 'duration_time',
        "words", "freqs",
        "gender", "beauty",
        "pos0", "pos1", "pos2", "pos3"]
target = ['finish', 'like']


#feature_range = {'uid': {'min':0, 'max':670000},
#                 'user_city', {'min':0, 'max':600},
#                 'item_id', {'min':0, 'max':32000000},

#sparse_feature_list = [SingleFeat('uid', 670000),
#                       SingleFeat('user_city', 600),
#                       SingleFeat('item_id', 32000000),
#                       SingleFeat('author_id', 15000000),
#                       SingleFeat('item_city', 500),
#                       SingleFeat('channel', 6),
#                       SingleFeat('music_id', 7800000),
#                       SingleFeat('device', 80000),
#                       ]
sparse_feature_list = [SingleFeat('uid', 80000),
                       SingleFeat('user_city', 400),
                       SingleFeat('item_id', 4200000),
                       SingleFeat('author_id', 860000),
                       SingleFeat('item_city', 470),
                       SingleFeat('channel', 6),
                       SingleFeat('music_id', 90000),
                       SingleFeat('device', 80000),
                       ]
dense_feature_list = [SingleFeat('duration_time', 0)]

ONLINE_FLAG = False
if ONLINE_FLAG:
    train_file = 'input/train.txt'
    test_file = 'input/test.txt'
    train_data_len = 10000
    epochs = 1
else:
    train_file = 'input/train.txt'
    test_file = 'input/test.txt'
    train_data_len = 10000
    epochs = 5

batch_size = 4096
train_steps_per_epoch = train_data_len // batch_size
duration_time_max = 150

sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'music_id', 'device']
dense_features = ['duration_time', ]
