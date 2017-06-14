import os
import sys
import numpy as np
import pickle
import h5py
from PIL import Image
from scipy import misc
from scipy import io as sio
import matplotlib.pyplot as plt


def get_image_num(path, ext):
    sizes = []
    for dir in os.listdir(path):
        sizes += [0]
        for root, dirs, files in os.walk(os.path.join(path, dir)):
            for file in files:
                if os.path.splitext(file)[1] == ext:
                    sizes[-1] += 1
    return sizes


def count_file(path,ext):
    size = 0
    for f in os.listdir(path):
        if os.path.splitext(f)[1] == ext:
            size += 1
    return size


def get_train_data(path, ext, size):
    """ Create the image db for training
    
    Input: image path, extension name, picture size=(w,h,c)
    
    The folder structure should be [lable_number]~[lable name] to distinguish class,
        eg. 0~abnormal, 1~normal
    Each image (in [path]/*.[ext]) will be load into imdb as well as its table.
    The mode of the image must be all the same.
        
    Output: x_train, y_train, info=('size','lable_name','avg_image')
            x_train, y_train are lists
    
    """
    s = get_image_num(path, ext)
    n = sum(s)
    output = len(s)

    x_train = np.empty((n,) + size, dtype='float32')
    y_train = np.zeros((n,output), dtype='float32')

    info = dict()
    info['image_count'] = s
    info['size'] = size
    info['avg_image'] = np.zeros(size, dtype='float32')
    info['image_path'] = []
    info['lable_name'] = []
    i = 0

    for dir in os.listdir(path):
        lable_num, lable_name = dir.split('~')
        info['lable_name'].append (lable_name)
        for root, dirs, files in os.walk(os.path.join(path, dir)):
            for file in files:
                f = os.path.join(root, file)
                if ext == os.path.splitext(f)[1]:
                    x_train[i] = load_image(f, resize=size)
                    y_train[i][lable_num] = 1.0
                    info['image_path'].append(str(f))
                    info['avg_image'] += x_train[i]
                    print(f + '  has been loaded')
                    i += 1
    info['avg_image'] /= i
    print('Getting training data successfully !')
    return x_train, y_train, info


def get_train_data2(path, ext, size):
    """ Create the image db2 for training

    Input: image path, extension name, picture size=(w,h,c)

    First it read lable_name.txt from path.
    lable_name.txt format: [lable_number]~[lable name]
                   eg. 0~abnormal, 1~normal (the number must be increased)
     
    Then it read all the folders in path, packing each folder in *.mat, and then load their pictures with ext extension name.
    The y_train is according to the tag.txt in each folder (if there is not, raise exception IOError)
    tag.txt format: (note that the filename should not include "[" or ']', the filename eg. 11.png, no dir path)
    [0]
    filename 
    ...
    [1]
    filename
    ...
    
    
    Output: x_train, y_train, info=('size','lable_name','avg_image')
            x_train, y_train are dicts

    """
    info = dict()
    info['image_count'] = 0
    info['size'] = size
    info['avg_image'] = np.zeros(size, dtype='float32')
    info['image_path'] = {}
    info['lable_name'] = []

    for line in open(os.path.join(path, 'lable_name.txt'), 'r').readlines():
        lable_num, lable_name = line.split('~')
        info['lable_name'].append(lable_name)

    x_train = {}
    y_train = {}
    for d in os.listdir(path):
        if not os.path.isdir(os.path.join(path,d)):
            continue
        i = 0
        n = count_file(os.path.join(path,d), ext)
        output = len(info['lable_name'])
        x_train[d] = np.empty((n,) + size, dtype='float32')
        y_train[d] = np.zeros((n, output), dtype='float32')
        info['image_path'][d] = []
        for line in open(os.path.join(path, d, 'tag.txt'), 'r').readlines():
            if line[0] == '[':
                tag = int(line.split('[')[1].split(']')[0])
            elif line[0] != ' ' and line[0] != '\n':
                line = line.split('\n')[0]
                x_train[d][i] = load_image(os.path.join(path, d, line), size)
                y_train[d][i][tag] = 1.0
                info['image_path'][d].append(line)
                info['avg_image'] += x_train[d][i]
                i += 1
                print(os.path.join(path, d, line) + '  has been loaded')
        info['image_count'] += i
    info['avg_image'] /= info['image_count']
    print('Getting training data successfully !')
    return x_train, y_train,info


def subavg_data(img, avg_img):
    for i in range(0, img.shape[0]):
        img[i] -= avg_img


def save_mat_data(path, varname, arr):
    sio.savemat(path, {varname: arr})


def load_mat_data(path, varname):
    arr = sio.loadmat(path)
    return arr[varname]


def save_train_data(path, x_train, y_train, info):
    sio.savemat(os.path.join(path, 'x_train.mat'),{'x_train':x_train})
    print("%s(%.3f mb) has been saved in %s" % ('x_train', sys.getsizeof(x_train) / 1000000, path))
    sio.savemat(os.path.join(path, 'y_train.mat'), {'y_train': y_train})
    print("%s(%.3f mb) has been saved in %s" % ('y_train', sys.getsizeof(y_train) / 1000000, path))
    pickle.dump(info, open(os.path.join(path, 'info.pkl'),'wb'))


def load_train_data(path):
    d = sio.loadmat(os.path.join(path, 'x_train.mat'))
    x_train = d['x_train']
    print("%s has been loaded in %s" % ('x_train', path))
    d = sio.loadmat(os.path.join(path, 'y_train.mat'))
    y_train = d['y_train']
    print("%s has been loaded in %s" % ('y_train', path))
    info = pickle.load(open(os.path.join(path, 'info.pkl'), 'rb'))
    return x_train, y_train, info


def load_image(path, resize=None):
    im = Image.open(path)
    if resize:
        im = misc.imresize(im, resize)
    arr = np.asarray(im, dtype='float32')
    if resize:
        arr = arr.reshape(resize)
    return arr


def save_image(path, arr, resize=None):
    if resize:
        arr = arr.reshape(resize)
    im = Image.fromarray(arr)
    misc.imsave(path, im)


if __name__ == '__main__':
    prex_train, prey_train, info=get_train_data2("D:\\ProcMake\\current\\ABD\\row\\rnn_uscd_biclassification", '.tif', (260, 260, 1))
    for i in prex_train:
        subavg_data(prex_train[i], info['avg_image'])
    # save_mat_data(os.path.join("D:\\ProcMake\\current\\ABD\\intermediate\\train_data", 'xavg_train.mat'),'xavg_train', x_train)
    arr = prex_train['Train001'][13]
    arr=info['avg_image']
    #avg = info['avg_image']
    save_image("D:\\ProcMake\\current\\ABD\\image\\22.png", arr, (260, 260))
