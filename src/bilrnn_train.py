"""
   to train a LTSM rnn (as well as GRU) that judge whether it is abnormal in the last frame of a video sequence 
   input: a sequence of frame feature vector trained by a fine-turned CNN
   padding: for the first several frames, just copy the first image for padding...
"""

from prepdata import *
from abd_model_ini import *
import keras
import time
from keras.models import Model
import random


data_exsit = True
path_lrnn_weight = None
path_cnn_weight = r"D:\ProcMake\current\ABD\intermediate\model\20170523_1733\bicnn_w42.h5py"
is_continue = False
ini_epoch = 0
move_step = 1


class SaveModelWeight(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        path = os.path.join(log_path, 'bilrnn_w' + str(epoch) + '.h5py')
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

# model_lrnn = create_bi_lrnn()
model_lrnn = create_bi_grnn() # to select GRU RNN
if data_exsit:
    x_train = load_mat_data(os.path.join(path_train_data, 'seqx_train.mat'), 'seqx_train')
    y_train = load_mat_data(os.path.join(path_train_data, 'seqy_train.mat'), 'seqy_train')
    info = pickle.load(open(os.path.join(path_train_data, 'seq_info.pkl'), 'rb'))
else:
    rowx_train, rowy_train, info = get_train_data2(path_train_bilrnn, img_ext, img_size)
    output = len(info['lable_name'])
    model_cnn = create_bi_cnn()
    model_cnn.load_weights(path_cnn_weight)
    fea_layer_model = Model(inputs=model_cnn.input,
                                 outputs=model_cnn.get_layer(index=14).output)
    if path_lrnn_weight:
        model_lrnn.load_weights(path_lrnn_weight)

    # count video clip number
    n = 0
    for name in rowy_train:
        subavg_data(rowx_train[name], info['avg_image'])
        n += int((len(rowx_train[name]) - vseq_len) / move_step + 1)
    x_train = np.zeros((n, vseq_len, fea_dim), dtype='float32')
    y_train = np.zeros((n, vseq_len,output), dtype='float32')

    # make rnn sequence
    j = 0
    for name in rowx_train:
        fea = fea_layer_model.predict(rowx_train[name])
        for i in range(vseq_len, len(rowx_train[name]), move_step):
            # y_train[j] = rowy_train[name][i]
            x_train[j] = fea[i-vseq_len:i]
            y_train[j] = rowy_train[name][i-vseq_len:i]
            j += 1
    save_mat_data(os.path.join(path_train_data, 'seqx_train.mat'), 'seqx_train', x_train)
    save_mat_data(os.path.join(path_train_data, 'seqy_train.mat'), 'seqy_train', y_train)
    pickle.dump(info, open(os.path.join(path_train_data, 'seq_info.pkl'), 'wb'))

shuffle_train_data(x_train, y_train, 3)
if path_lrnn_weight and is_continue:
    log_path = os.path.split(path_lrnn_weight)[0]
    ini_epoch = int(path_lrnn_weight.split('_w')[-1].split('.h5py')[0]) + 1
else:
    tstr = time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))
    log_path = os.path.join(path_train_model, tstr)
    os.makedirs(log_path)
model_lrnn.fit(x_train, y_train,
                   batch_size=128, epochs=200, initial_epoch=ini_epoch,
                   validation_split=0.25,
                   callbacks=[SaveModelWeight()])