import os
from deepctr import SingleFeat
from model import FeatureInfo, EmbeddingFeatureInfo, NumericFeatureInfo

ngpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

ONLINE_FLAG = True
if ONLINE_FLAG:
    train_file = '/srv/nbs/1/track1/origin/xa'
    eval_file = '/srv/nbs/1/track1/origin/xb'
    test_file = '/srv/nbs/1/track1/origin/final_track1_test_no_anwser.txt'
    train_data_len = 240000000
    epochs = 1
else:
    train_file = '/srv/nbs/1/track1/origin/xa'
    eval_file = '/srv/nbs/1/track1/origin/xb'
    test_file = '/srv/nbs/1/track1/origin/final_track1_test_no_anwser.txt'
    train_data_len = 240000000
    epochs = 3
batch_size = 4096
train_steps_per_epoch = train_data_len // batch_size
embedding_size = 8

# data schema
column_names = [
    'uid',
    'user_city',
    'item_id',
    'author_id',
    'item_city',
    'channel',
    'finish',
    'like',
    'music_id',
    'device',
    'creat_time',
    'duration_time',
]


# features will be used
features = [
    "uid",
    "item_id",
    "author_id",
    "item_city", 
    "channel",
    "music_id",
    "device",
    "duration_time",
    ]
targets = ['finish', 'like']


# feature conf for model
input_feature_list = [FeatureInfo(feature, "string", 1) for feature in features]

embedding_feature_list = [
    EmbeddingFeatureInfo('uid', 660000, dimension=embedding_size),
    EmbeddingFeatureInfo('item_id', 30000000, dimension=embedding_size),
    EmbeddingFeatureInfo('author_id', 16000000, dimension=embedding_size),
    EmbeddingFeatureInfo('item_city', 500, dimension=embedding_size),
    EmbeddingFeatureInfo('channel', 6, dimension=embedding_size),
    EmbeddingFeatureInfo('music_id', 7000000, dimension=embedding_size),
    EmbeddingFeatureInfo('device', 5000, dimension=embedding_size),
    #EmbeddingFeatureInfo('_X_'.join(['uid', 'item_id']), 100000, dimension=embedding_size),
    #EmbeddingFeatureInfo('_X_'.join(['uid', 'author_id']), 100000, dimension=embedding_size),
    #EmbeddingFeatureInfo('_X_'.join(['uid', 'channel']), 100000, dimension=embedding_size),
]

numeric_feature_list = [
    NumericFeatureInfo('duration_time', 1),
]




