from prepdata import *
from abd_model_ini import *
from keras.models import Model
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model


def demo_bicnn():
    path_weight = path_train_model + "/20170511_1057/bicnn_w21.h5py"
    model = create_bi_cnn()
    model.load_weights(path_weight)
    x_train = load_mat_data(os.path.join(path_train_data, 'xavg_train.mat'), 'xavg_train')
    y_train = load_mat_data(os.path.join(path_train_data, 'y_train.mat'), 'y_train')
    info = pickle.load(open(os.path.join(path_train_data, 'info.pkl'), 'rb'))
    e = model.predict_classes(x_train, 256, verbose=1)
    print(e)


def demo_bilrnn():
    path_weight = path_train_model + "/seq=20,src_0.3.1,128,noshuffle/20170514_2126/bilrnn_w199.h5py"
    model = create_bi_lrnn()
    model.load_weights(path_weight)
    x_train = load_mat_data(os.path.join(path_train_data, 'seqx_train.mat'), 'seqx_train')
    y_train = load_mat_data(os.path.join(path_train_data, 'seqy_train.mat'), 'seqy_train')
    info = pickle.load(open(os.path.join(path_train_data, 'seq_info.pkl'), 'rb'))
    e = model.predict_classes(x_train, 256, verbose=1)
    ee = model.evaluate(x_train, y_train,128,verbose=1)
    print(e)

def demo_model(model):
    plot_model(model, to_file=os.path.join("../image/",model.name + ".png"), show_shapes=True)
    pass


def read_train_log(path):
    """
    :param path: 
    :return: (test_loss, test_acc, val_loss, val_acc) in str format
    """
    arr_str = []
    for line in open(path, 'r').readlines():
        if  line.find('val_loss')==-1:
            continue
        item = []
        item.append(line.split('- loss: ')[1].split(' -')[0])
        item.append(line.split('- acc: ')[1].split(' -')[0])
        item.append(line.split('- val_loss: ')[1].split(' -')[0])
        item.append(line.split('- val_acc: ')[1].split('\n')[0])
        arr_str.append(item)
    return arr_str


if __name__ == '__main__':
    # demo_bilrnn()
    # demo_model(create_bi_lrnn())
    arr_str = read_train_log(r"D:\ProcMake\current\ABD\intermediate\model\20170523_2017_GRU\LOG.txt")
    for str in arr_str:
        print(str[0] + "\t" + str[1] + "\t" + str[2] + "\t" + str[3])