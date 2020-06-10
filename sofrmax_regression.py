"""softmax回归"""
from mxnet.gluon import data as gdata
import sys
import time
import Myself_d2lzh as my
from mxnet import nd,autograd
mnist_train = gdata.vision.FashionMNIST(train=True)#下载FashionMNIST数据集
mnist_test = gdata.vision.FashionMNIST(train=False)

feature,label=mnist_train[0]
print('feature.shape',feature.shape)
train_iter,test_iter = my.load_data_fashion_mnist(256)
for X,y in test_iter:
    print(".....",X.shape)

# print(feature)
# print('feature.shape',feature.shape)
# # print('feature.dtype',feature.dtype)
# # print(feature.dtype)
# # x,y=mnist_train[0:18]
# # my.show_fashion_mnist(x,my.get_fashion_mnist_labels(y))
# # print("over")

# def softmax(x):
#     x_exp = x.exp()
#     partition = x_exp.sum(axis=1,keepdims=True)#axis=1对一列求和，axis=0对一行求和
#     return x_exp/partition
# def net(x):
#     return softmax(nd.dot(x.reshape(-1,num_inputs),w)+b)
# def cross_entropy(y_hat,y):
#     a = -nd.pick(y_hat, y).log()
#     return a
# def accuracy(y_hat, y):
#     return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()
# # 整个数据集
# def evaluate_accuracy(data_iter, net):
#     acc = 0
#     for X, y in data_iter:
#         acc += accuracy(net(X), y)
#     return acc / len(data_iter)
#
#
# # 本函数已保存在 my 包中方便以后使用。
# def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
#               params=None, lr=None, trainer=None):
#     for epoch in range(num_epochs):
#         train_l_sum = 0
#         train_acc_sum = 0
#         for X, y in train_iter:
#             with autograd.record():
#                 y_hat = net(X)
#                 l = loss(y_hat, y)
#             l.backward()
#             if trainer is None:
#                 my.sgd(params, lr, batch_size)
#             else:
#                 trainer.step(batch_size)  # 下一节将用到。
#             train_l_sum += l.mean().asscalar()#mean()求均值 asscalar()将向量转化为标量
#             train_acc_sum += accuracy(y_hat, y)
#         test_acc = evaluate_accuracy(test_iter, net)
#         print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
#               % (epoch + 1, train_l_sum / len(train_iter),
#                  train_acc_sum / len(train_iter), test_acc))
#
#
# batch_size = 256
# train_iter,test_iter = my.load_data_fashion_mnist(batch_size)
# print(len(train_iter))
# num_inputs = 784
# num_outputs = 10
# w = nd.random.normal(scale=0.01,shape=(num_inputs,num_outputs))
# b = nd.zeros(num_outputs)
# w.attach_grad()
# b.attach_grad()
# num_epochs, lr = 5, 0.1
# train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs,
#           batch_size, [w, b], lr)
#
# print('w:', w)
# print('b:', b)
#
#
# for X, y in test_iter:
#     break
# true_labels = my.get_fashion_mnist_labels(y.asnumpy())
# pred_labels = my.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
# titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
#
# my.show_fashion_mnist(X[0:9], titles[0:9])

