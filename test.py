

column_names = ["instance_id", 'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'device', 'creat_time', 'duration_time',
        "words", "freqs",
        "gender", "beauty",
        "pos0", "pos1", "pos2", "pos3"]

file_name = 'input/train.txt'
df = pd.read_csv(file_name, sep='\t', names=column_names, header=None, nrows=100)
