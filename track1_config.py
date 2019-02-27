from deepctr import SingleFeat

column_names = ["instance_id", 'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'device', 'creat_time', 'duration_time',
        "words", "freqs",
        "gender", "beauty",
        "pos0", "pos1", "pos2", "pos3"]
target = ['finish', 'like']

#sparse_feature_list = [SingleFeat('uid', 670000),
#                       SingleFeat('item_id', 32000000),
#                       SingleFeat('author_id', 15000000),
#                       SingleFeat('item_city', 500),
#                       SingleFeat('channel', 6),
#                       SingleFeat('music_id', 7800000),
#                       SingleFeat('device', 5000),
#                       ]
sparse_feature_list = [SingleFeat('uid', 670000),
                       SingleFeat('item_id', 320000),
                       SingleFeat('author_id', 150000),
                       SingleFeat('item_city', 500),
                       SingleFeat('channel', 6),
                       SingleFeat('music_id', 780000),
                       SingleFeat('device', 5000),
                       ]
dense_feature_list = [SingleFeat('duration_time', 0)]
    
ONLINE_FLAG = False
if ONLINE_FLAG:
    train_file = '/srv/nbs/1/track1/final_track1_train_new.txt'
    test_file = '/srv/nbs/1/track1/final_track1_test_new.txt'
    train_data_len = 275855531
    epochs = 1
else:
    train_file = '/srv/nbs/1/track1/xa'
    test_file = '/srv/nbs/1/track1/xb'
    train_data_len = 240000000
    epochs = 5

batch_size=4096
train_steps_per_epoch = train_data_len // batch_size
duration_time_max = 150

sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'music_id', 'device']
dense_features = ['duration_time', ]
