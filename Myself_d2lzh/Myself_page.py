import os
import sys
import random
from IPython import display
from mxnet import nd,autograd
from matplotlib import pyplot as plt
from mxnet.gluon import data as gdata



#线性模型
def linreg(X,w,b):
    return nd.dot(X,w)+b


#定义损失函数
def squared_loss(y_hat,y):
    return (y_hat - y.reshape(y_hat.shape))**2/2#shape读取矩阵长度 reshape返回一个矩阵
def set_figsize(figsize=(3.5,2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize']=figsize
####小批量读取数据集
def data_iter(batch_size, features, labels):
    """Iterate through a data set."""
    num_examples = len(features)#列表长度
    indices = list(range(num_examples)) #生成一个num_example长度的顺序列表
                                        #input: print(list(range(10)))
                                        #output: [0,1,2,3,4,5,6,7,8,9]
    random.shuffle(indices) #打乱顺序
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)

#定义优化算法
def sgd(params,lr,batch_size):
    for param in params:
        param[:] = param-lr*param.grad/batch_size
def get_fashion_mnist_labels(labels): #fashion_mnist数据集的文字标签
    """Get text label for fashion mnist."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

    return [text_labels[int(i)] for i in labels]
    #'t-shirt()', 'trouser（裤子）', 'pullover（套衫）', 'dress', 'coat',/
    # 'sandal（凉鞋）', 'shirt', 'sneaker（运动鞋）', 'bag', 'ankle boot（短靴）'

def get_mnist_labels(labels):
    text_labels = ['0', '1', '2', '3', '4','5', '6', '7', '8', '9']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images, labels):
    """Plot Fashion-MNIST images with labels."""
    use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()
def use_svg_display():#用矢量图显示
    display.set_matplotlib_formats("svg")
def load_data_fashion_mnist(batch_size, resize=None, root=os.path.join(
        '~', '.mxnet', 'datasets', 'fashion-mnist')):
    """Download the fashion mnist dataset and then load into memory."""
    print("root---before:",root)
    root = os.path.expanduser(root)   #返回数据集的地址
    print("root---after:", root)
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)

    mnist_train = gdata.vision.FashionMNIST(root=root, train=True)#下载fashion_mnist数据集
    mnist_test = gdata.vision.FashionMNIST(root=root, train=False)
    print('mnist_train的大小 %d' % (len(mnist_train)))
    num_workers = 0 if sys.platform.startswith('win32') else 4
    print('num_workers %d' % (num_workers))
    train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                                  batch_size, shuffle=True,
                                  num_workers=num_workers)
    test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                                 batch_size, shuffle=False,
                                 num_workers=num_workers)
    return train_iter, test_iter
def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()
# 整个数据集
def evaluate_accuracy(data_iter, net):
    acc = 0
    for X, y in data_iter:
        acc += accuracy(net(X), y)
    return acc / len(data_iter)
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum = 0
        train_acc_sum = 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            if trainer is None:
                sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)  # 下一节将用到。
            train_l_sum += l.mean().asscalar()#mean()求均值 asscalar()将向量转化为标量
            train_acc_sum += accuracy(y_hat, y)
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / len(train_iter),
                 train_acc_sum / len(train_iter), test_acc))
def semilogy(lanbd,x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    """Plot x and log(y)."""
    set_figsize(figsize)
    string = "$\lambda$="+str(lanbd)
    plt.title(string)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals,y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()