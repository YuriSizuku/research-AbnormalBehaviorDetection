from prepdata import *
from abd_model_ini import *
import keras
import time

data_exsit = True
path_weight = path_train_model + "/20170405_1106/bicnn_w1.h5py"
is_continue = True
ini_epoch = 0


class SaveModelWeight(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        path = os.path.join(log_path, 'bicnn_w' + str(epoch) + '.h5py')
        self.model.save_weights(path)


if data_exsit:
    x_train = load_mat_data(os.path.join(path_train_data, 'xavg_train.mat'), 'xavg_train')
    y_train = load_mat_data(os.path.join(path_train_data, 'y_train.mat'), 'y_train')
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
his = model.fit(x_train, y_train,
                batch_size=128, epochs=2, validation_split=0.25,
                shuffle=True, initial_epoch=ini_epoch,
                callbacks=[SaveModelWeight()])
# print(his, file=open(os.path.join(log_path, 'log.txt'), 'w+'))
