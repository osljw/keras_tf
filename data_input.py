import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras


#class DataGenerator(keras.utils.Sequence):
#    def __init__(batch_size):
#        self.batch_size = batch_size
#        self.steps = int(np.floor(len(self.list_IDs) / self.batch_size))
#
#    def __len__(self):
#        return self.steps
#
#    def __getitem__(self, index):
#        pass


# data preprocess
duration_time_max = 150

def duration_min_max(x):
    return (x-0)/(duration_time_max - 0)

def data_clip(x):
    return x if x > 0 else 1

def data_preprocess(df):
    df['duration_time'] = df['duration_time'].apply(duration_min_max)
    df = df.fillna(0)
    return df

print_head = True
def data_generator(file_names, column_names, features, targets, batch_size, epochs=1):
    global print_head
    file_names = tf.gfile.Glob(file_names)
    while epochs:
        for file_name in file_names:
            print("read file:{}".format(file_name))
            fd = open(file_name)
            reader = pd.read_csv(fd, sep='\t', chunksize=batch_size, names=column_names, header=None)
            for chunk_df in reader:
                #print("dtypes:", chunk_df.dtypes)
                #print("data before modify:", chunk_df['user_city'].head())
                chunk_df = data_preprocess(chunk_df)
                #print("data after modify:", df['user_city'].head())
                X = {feature: chunk_df[feature].astype(str).values for feature in features}
                Y = {target:chunk_df[target].values for target in targets}
                if print_head == True:
                    print("columns:", list(chunk_df.columns))
                    print(chunk_df.head())
                    print("X", X)
                    print_head = False
                #print("data rows:{}".format(len(chunk_df)))
                yield X, Y
            fd.close()
        epochs -= 1
