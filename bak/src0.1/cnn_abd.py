import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, Dropout
from keras.optimizers import SGD


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
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model


def save_bi_cnn(path,model):
    pass

def load_bi_cnn(path):
    pass

if __name__ == '__main__':
    model=create_bi_cnn()
    print(model)