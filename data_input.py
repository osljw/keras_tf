import numpy as np
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


def data_preprocess(df):
    #print("sparse_features", sparse_features)
    df['duration_time'] = df['duration_time'].apply(duration_min_max)
    df[sparse_features] = df[sparse_features].apply(lambda x: x.clip(lower=0))
    df[dense_features] = df[dense_features].fillna(0,)
    return df

def data_generator(file_name):
    fd = open(file_name)
    reader = pd.read_csv(fd, sep='\t', chunksize=batch_size, names=column_names, header=None)
    while True:
        for chunk_df in reader:
            #print("dtypes:", chunk_df.dtypes)
            #print("data before modify:", chunk_df['user_city'].head())
            chunk_df = data_preprocess(chunk_df)
            #print("data after modify:", df['user_city'].head())
            X = [chunk_df[feat.name].values for feat in sparse_feature_list] + \
                    [chunk_df[feat.name].values for feat in dense_feature_list]
            Y = [chunk_df[target[0]].values, chunk_df[target[1]].values]
            yield X, Y
        fd.close()
        fd = open(file_name)
        reader = pd.read_csv(fd, sep='\t', chunksize=batch_size, names=column_names, header=None)

def data_generator_new(file_name):
    fd = open(file_name)
    reader = pd.read_csv(fd, sep='\t', chunksize=batch_size, names=column_names, header=None)
    while True:
        for chunk_df in reader:
            #print("dtypes:", chunk_df.dtypes)
            #print("data before modify:", chunk_df['user_city'].head())
            chunk_df = data_preprocess(chunk_df)
            #print("data after modify:", df['user_city'].head())
            X = [chunk_df[feat.name].values for feat in sparse_feature_list] + \
                    [chunk_df[feat.name].values for feat in dense_feature_list]
            #Y = [chunk_df[target[0]].values, chunk_df[target[1]].values]
            Y = {"finish":chunk_df[target[0]].values, "like":chunk_df[target[1]].values}
            yield X, Y
        fd.close()
        fd = open(file_name)
        reader = pd.read_csv(fd, sep='\t', chunksize=batch_size, names=column_names, header=None)
