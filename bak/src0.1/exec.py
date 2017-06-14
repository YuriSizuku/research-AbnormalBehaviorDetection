from prepdata import *
from abd_model_ini import *
from keras.callbacks import ModelCheckpoint

# ini path
path_row='../row'
path_data_bicnn= os.path.join(path_row, 'cnn_uscd_biclassification')
path_train_model='../intermediate/model'
path_test_model='../model'
path_train_data='../intermediate/train_data'
train_files = ['x_train.mat', 'y_train.mat', 'info.pkl']

# para
img_size =(260, 260, 1)
img_ext = '.tif'
data_exsit = 1


def train_bicnn(x_train, y_train, info, model):
    model.fit(x_train, y_train,
              batch_size=128, epochs=20, validation_split=0.25,
              shuffle=True,
              callbacks=[ModelCheckpoint(path_train_model+"/weights.hdf5")])


def test_bicnn(model):
    pass


if __name__ == '__main__':
    if data_exsit == 1:
        x_train, y_train, info = load_train_data2(path_train_data)
    else:
        for f in train_files:
            if os.path.isfile(os.path.join(path_train_data, f)) == 0:
                data_exsit = 0
                x_train, y_train, info = get_train_data(path_data_bicnn, img_ext, img_size)
                break
        save_train_data(path_train_data, x_train, y_train, info)
    model = create_bi_cnn()
    train_bicnn(x_train, y_train, info, model)