import os
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, Dropout
from keras.layers import LSTM
from keras.optimizers import SGD


# ini path
path_row = '../row'
path_data_bicnn = os.path.join(path_row, 'cnn_uscd_biclassification')
path_train_bilrnn = os.path.join(path_row, 'rnn_uscd_biclassification')
path_train_model = '../intermediate/model'
path_test_model = '../model'
path_train_data = '../intermediate/train_data'

# para
img_size =(260, 260, 1)
img_ext = '.tif'
fea_dim = 256  # the dimension of feature vector
vseq_len = 20  # the frame count of the video


def create_bi_cnn():
    model = Sequential()
    # 1
    model.add(Conv2D(86, (5, 5), strides=3, activation = 'relu', input_shape=(260, 260, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    # 2
    model.add(ZeroPadding2D(1))
    model.add(Conv2D(233, (5, 5), strides=2, activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    # 3
    model.add(ZeroPadding2D(1))
    model.add(Conv2D(332, (3, 3), activation = 'relu'))
    # 4
    model.add(ZeroPadding2D(1))
    model.add(Conv2D(332, (3, 3), activation = 'relu'))
    # 5
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    # 6
    model.add(Conv2D(256, (4, 4), activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    # 7
    model.add(Dense(fea_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model


def create_bi_lrnn():
    model = Sequential()

    model.add(LSTM(128, input_shape=(vseq_len, fea_dim), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model


def save_bi_cnn(path,model):
    pass


def load_bi_cnn(path):
    pass


if __name__ == '__main__':
    model=create_bi_lrnn()
    print(model)