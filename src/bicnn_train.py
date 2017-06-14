from prepdata import *
from abd_model_ini import *
import keras
import time
import random

data_exsit = True
path_weight = None #"D:\\ProcMake\\current\\ABD\\intermediate\\model\\20170510_1523\\bicnn_w199.h5py"
is_continue = False
ini_epoch = 0


class SaveModelWeight(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        path = os.path.join(log_path, 'bicnn_w' + str(epoch) + '.h5py')
        self.model.save_weights(path)


def shuffle_train_data(x_train, y_train, round=1):
    n = y_train.shape[0]
    for var in range(0, round):
        for i in range(0, n-1):
            ii = random.randint(0, n-1)
            tmp = x_train[i]
            x_train[i] = x_train[ii]
            x_train[ii] = tmp
            tmp = y_train[i]
            y_train[i] = y_train[ii]
            y_train[ii] = tmp


if data_exsit:
    x_train = load_mat_data(os.path.join(path_train_data, 'xavg_train.mat'), 'xavg_train')
    y_train = load_mat_data(os.path.join(path_train_data, 'y_train.mat'), 'y_train')
    info = pickle.load(open(os.path.join(path_train_data, 'info.pkl'), 'rb'))
else:
    x_train, y_train, info = get_train_data(path_data_bicnn, img_ext, img_size)
    save_train_data(path_train_data, x_train, y_train, info)
    subavg_data(x_train, info['avg_image'])
    save_mat_data(os.path.join(path_train_data, 'xavg_train.mat'), 'xavg_train', x_train)
if path_weight and is_continue:
    log_path = os.path.split(path_weight)[0]
    ini_epoch = int(path_weight.split('_w')[-1].split('.h5py')[0]) + 1
else:
    tstr = time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))
    log_path = os.path.join(path_train_model, tstr)
    os.makedirs(log_path)

model = create_bi_cnn()
if path_weight:
    model.load_weights(path_weight)
shuffle_train_data(x_train, y_train, 3)
his = model.fit(x_train, y_train,
                batch_size=128, epochs=200, validation_split=0.2,
                shuffle=True, initial_epoch=ini_epoch,
                callbacks=[SaveModelWeight()])
#print(his, file=open(os.path.join(log_path, 'log.txt'), 'w+'))
