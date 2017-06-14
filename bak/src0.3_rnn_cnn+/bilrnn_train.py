"""
   to train a LTSM rnn that judge whether it is abnormal in the last frame of a video sequence 
   input: a sequence of frame feature vector trained by a fine-turned CNN
   padding: for the first several frames, just copy the first image for padding...
"""

from prepdata import *
from abd_model_ini import *
import keras
import time
from keras.models import Model


data_exsit = False
path_lrnn_weight = None
path_cnn_weight = "D:\\ProcMake\\current\\ABD\\intermediate\\model\\20170510_1523\\bicnn_w199.h5py"
is_continue = False
ini_epoch = 0
move_step = 1


class SaveModelWeight(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        path = os.path.join(log_path, 'bilrnn_w' + str(epoch) + '.h5py')
        self.model.save_weights(path)


if path_lrnn_weight and is_continue:
    log_path = os.path.split(path_lrnn_weight)[0]
    ini_epoch = int(path_lrnn_weight.split('_w')[-1].split('.h5py')[0]) + 1
else:
    tstr = time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))
    log_path = os.path.join(path_train_model, tstr)
    os.makedirs(log_path)

if data_exsit:  # I don't want to do this part now...
    exit(1)
else:
    rowx_train, rowy_train, info = get_train_data2(path_train_bilrnn, img_ext, img_size)
    output = len(info['lable_name'])
    model_lrnn = create_bi_lrnn()
    model_cnn = create_bi_cnn()
    model_cnn.load_weights(path_cnn_weight)
    l15 = model_cnn.get_layer(index=14)
    fea_layer_model = Model(inputs=model_cnn.input,
                                 outputs=model_cnn.get_layer(index=14).output)
    if path_lrnn_weight:
        model_lrnn.load_weights(path_lrnn_weight)
    for name in rowx_train:
        # sub average image
        subavg_data(rowx_train[name], info['avg_image'])
        # cnn extract feature
        fea = []
        # no padding
        n = len(rowx_train[name])
        x_train = np.zeros(((n-vseq_len)/move_step + 1, vseq_len, fea_dim), dtype='float32')
        y_train = np.zeros(((n-vseq_len)/move_step + 1, output), dtype='float32')
        fea = fea_layer_model.predict(rowx_train[name])
        j = 0
        for i in range(vseq_len, n,move_step):
            y_train[j] = rowy_train[i]
            x_train[j] = fea[i-vseq_len:i]
            j += 1
        model_lrnn.fit(x_train, y_train,
                  batch_size=128, epochs=1, initial_epoch=ini_epoch)