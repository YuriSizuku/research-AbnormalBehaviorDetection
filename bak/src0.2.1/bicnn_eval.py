from prepdata import *
from abd_model_ini import *
import keras

path_weight = path_train_model + "/0.2_20_weights.hdf5"


model = create_bi_cnn()
model.load_weights(path_weight)
x_train = load_mat_data(os.path.join(path_train_data, 'xavg_train.mat'), 'xavg_train')
y_train = load_mat_data(os.path.join(path_train_data, 'y_train.mat'), 'y_train')
# e = model.evaluate(x_train, y_train, batch_size=128)
e = model.predict_classes(x_train, 256, verbose=1)
print(e)