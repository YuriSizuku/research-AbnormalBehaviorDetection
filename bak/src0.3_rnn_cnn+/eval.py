from prepdata import *
from abd_model_ini import *
from keras.models import Model
import keras

path_weight = path_train_model + "/20170511_1057/bicnn_w21.h5py"


model = create_bi_cnn()
model.load_weights(path_weight)
x_train = load_mat_data(os.path.join(path_train_data, 'xavg_train.mat'), 'xavg_train')
y_train = load_mat_data(os.path.join(path_train_data, 'y_train.mat'), 'y_train')
info = pickle.load(open(os.path.join(path_train_data, 'info.pkl'), 'rb'))
#e1 = model.evaluate(x_train, y_train, batch_size=128)
w = model.get_weights()
l15 = model.get_layer(index=14)
fea_layer_model = Model(inputs=model.input,
                        outputs=model.get_layer(index=14).output)
ee = fea_layer_model.predict(x_train)
e3 = model.predict(x_train, 256, verbose=1)
e2 = model.predict_classes(x_train, 256, verbose=1)
print()