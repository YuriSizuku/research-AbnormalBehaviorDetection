from prepdata import *
from abd_model_ini import *
import keras
import time

data_exsit = 1
path_weight = ""

tstr = time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))
log_path = os.path.join(path_train_model, tstr)
os.makedirs(log_path)


class SaveModelWeight(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        path = os.path.join(log_path, 'bicnn_w' + str(epoch) + '.h5py')
        self.model.save_weights(path)


if data_exsit == 1:
    x_train = load_mat_data(os.path.join(path_train_data, 'xavg_train.mat'), 'xavg_train')
    y_train = load_mat_data(os.path.join(path_train_data, 'y_train.mat'), 'y_train')
else:
    x_train, y_train, info = get_train_data(path_data_bicnn, img_ext, img_size)
    save_train_data(path_train_data, x_train, y_train, info)
    subavg_data(x_train, info['avg_image'])
    save_mat_data(os.path.join(path_train_data, 'xavg_train.mat'), 'xavg_train', x_train)

model = create_bi_cnn()
if path_weight:
    model.load_weights(path_weight)
    model.save_weights()
his = model.fit(x_train, y_train,
                batch_size=128, epochs=2, validation_split=0.25,
                shuffle=True,
                callbacks=[SaveModelWeight()])
print(his, file=open(os.path.join(log_path, 'log.txt'), 'w+'))
