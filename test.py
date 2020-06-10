from cgi import log

from mxnet import nd,autograd
from mxnet.gluon import data as gdata
import random
true_w = [2,-3.4]
true_b = 4.2
# features = nd.random.normal(scale=1,shape=(20,2))
# labels = true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b
# labels+=nd.random.normal(scale=0.01,shape=labels.shape)
# ####其中features是训练数据特征，labels是标签(y=Xw+b+噪声 噪声服从均值为0 标准差为0.01的正态分布)。
#
# w = nd.random.normal(scale=0.01, shape=(2, 1))  # 定义权重 偏置
# b = nd.zeros(shape=(1,))
# def data_iter(batch_size, features, labels):
#     """Iterate through a data set."""
#     num_examples = len(features)#列表长度
#     indices = list(range(num_examples)) #生成一个num_example长度的顺序列表
#                                         #input: print(list(range(10)))
#                                         #output: [0,1,2,3,4,5,6,7,8,9]
#     random.shuffle(indices) #打乱顺序
#     for i in range(0, num_examples, batch_size):
#         j = nd.array(indices[i: min(i + batch_size, num_examples)])
#         print(j)
#         print(j[0]+1)
#         print()
#         yield features.take(j), labels.take(j)
# #print(w)
# #print(nd.dot(features,w))
# for a,b in data_iter(10, features, labels):
#     print(a)
#     print(b)

# def softmax(x):
#     x_exp = x.exp()
#     print("x_exp %f" % (x_exp))
#     partition = x_exp.sum(axis=1,keepdims=True)#axis=1对一列求和，axis=0对一行求和
#     print("partition %f" % partition)
#     return x_exp/partition
# X = nd.random.normal(shape=(2,5))
# print(X)
# print(softmax(X))

# def cross_entropy(y_hat,y):
#     return nd.pick(y_hat,y).log()
# # x=nd.array([2])
# # print(cross_entropy(2,3))
# # print(log(2,3))
# y_hat = nd.array([[0.1,0.2,0.7], [0.3,0.2,0.5]])
# y = nd.array([0,2])
# print(y_hat)
#
# print(y_hat.log())
# print(nd.pick(y_hat, y))
# y = nd.array([0,0])
# print(nd.pick(y_hat, y))
# print(nd.pick(y_hat, y).log())   #2.4317214

# a=nd.array([1,2,3,4,5,6])
# size=1
# b=gdata.DataLoader(a,size)
# print(b)
# print(a)
# print(a.reshape(-1,2)) #-1就是不管行数，控制列数
# b = a.reshape(2,3)
# print(a)
# print(b)
# w=nd.random.normal(scale=0.01,shape=(784,10))
# print(w)
# print(w.shape)
#
# print((nd.zeros(10)).shape)
# print(nd.zeros(10))
K = nd.array([[1, -1]])
print(K)

"""
[
 [1,1,0,0,0,0,1,1]
 [1,1,0,0,0,0,1,1] 
 [1,1,0,0,0,0,1,1]
 [1,1,0,0,0,0,1,1]
 [1,1,0,0,0,0,1,1]
 [1,1,0,0,0,0,1,1]
]
高和宽分别为6像素和8像素的图像
中间4列为黑（0），其余为白（1）

[
[ 0,1,0,0,0,-1,0]
[ 0,1,0,0,0,-1,0]
[ 0,1,0,0,0,-1,0]
[ 0,1,0,0,0,-1,0]
[ 0,1,0,0,0,-1,0]
[ 0,1,0,0,0,-1,0]
]
"""

