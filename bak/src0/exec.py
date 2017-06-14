from prepdata import *
from abd_model_ini import *

# ini path
path_row='../row'
path_train_model='../intermediate/model'
path_test_model='../model'
path_train_data='../intermediate/train_data'


def train_bicnn():
    pass


def test_bicnn():
    pass


if __name__ == '__main__':
    # x_train, y_train, info=get_train_data("D:\\Study\\BIT\\学习内容\\4-2\\AbnormalDetection\\row\\cnn_uscd_biclassification",'.tif',(256, 256, 1),is_resize=1)
    # save_train_data("D:\\ProcMake\\current\\ABD\\train_data",x_train,y_train,info)
    # x_train, y_train, info = load_train_data("D:\\ProcMake\\current\\ABD\\intermediate\\train_data")
    # avg=info['avg_image']
    # arr=x_train[0]
    # save_image("D:\\ProcMake\\current\\ABD\\image\\1.png", arr, (256, 256))
    #print(os.walk(path_train_data))
    a = np.mat('1,2,3;4,5,6')
    b = np.array([[1, 1, 1], [2, 2, 2]])
    io.savemat('a.mat', {'matrix': a})
    io.loadmat('a.mat', {'matrix': b})
    print(b)
    pass