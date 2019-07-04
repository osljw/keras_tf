import os
from deepctr import SingleFeat
from model import FeatureInfo, EmbeddingFeatureInfo, NumericFeatureInfo

ngpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

ONLINE_FLAG = True
if ONLINE_FLAG:
    train_file = 'input/train.txt'
    eval_file = 'input/test.txt'
    test_file = 'input/test.txt'
    train_data_len = 10000
    epochs = 1
else:
    #train_file = 'input/train.txt'
    train_file = 'input/train/part-*'
    eval_file = 'input/test.txt'
    test_file = 'input/test.txt'
    train_data_len = 10000
    epochs = 5
batch_size = 4096
train_steps_per_epoch = train_data_len // batch_size
embedding_size = 8

# data schema
column_names = [
    "instance_id", 
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
    "words", 
    "freqs",
    "gender", 
    "beauty",
    "pos0", 
    "pos1", 
    "pos2", 
    "pos3"]


# features will be used
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
targets = ['finish', 'like']


# feature conf for model
input_feature_list = [FeatureInfo(feature, "string", 1) for feature in features]

embedding_feature_list = [
    EmbeddingFeatureInfo('uid', 100000, dimension=embedding_size),
    EmbeddingFeatureInfo('item_id', 100000, dimension=embedding_size),
    EmbeddingFeatureInfo('author_id', 100000, dimension=embedding_size),
    EmbeddingFeatureInfo('item_city', 100000, dimension=embedding_size),
    EmbeddingFeatureInfo('channel', 100000, dimension=embedding_size),
    EmbeddingFeatureInfo('music_id', 100000, dimension=embedding_size),
    EmbeddingFeatureInfo('words', 100000, dimension=embedding_size),
    EmbeddingFeatureInfo('_X_'.join(['uid', 'item_id']), 100000, dimension=embedding_size),
    EmbeddingFeatureInfo('_X_'.join(['uid', 'author_id']), 100000, dimension=embedding_size),
    EmbeddingFeatureInfo('_X_'.join(['uid', 'channel']), 100000, dimension=embedding_size),
    EmbeddingFeatureInfo('_X_'.join(['uid', 'words']), 100000, dimension=embedding_size),
]

numeric_feature_list = [
    NumericFeatureInfo('duration_time', 1),
]




