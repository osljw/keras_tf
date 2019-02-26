from deepctr import SingleFeat

column_names = ["instance_id", 'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'device', 'creat_time', 'duration_time',
        "words", "freqs",
        "gender", "beauty",
        "pos0", "pos1", "pos2", "pos3"]
target = ['finish', 'like']

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
    train_file = '../track2/final_track2_train_new.txt'
    test_file = '../track2/final_track2_test_new.txt'
    train_data_len = 275855531
else:
    train_file = 'input/train.txt'
    test_file = 'input/test.txt'
    train_data_len = 10000

batch_size = 4096
train_steps_per_epoch = train_data_len // batch_size
duration_time_max = 150
