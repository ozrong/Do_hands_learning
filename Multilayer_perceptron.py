"""多层感知机"""
import Myself_d2lzh as my
from mxnet import nd
from mxnet.gluon import loss as gloss


#激活函数Relu
def relu(x):
    return nd.maximum(x,0)

#定义模型
def net(x):
    X = x.reshape((-1,num_input))  #shape是查看数据有多少行多少列,reshape()作用是将数据重新组织
                                  #a=nd.array([1,2,3,4,5,6])
                                  #print(a)
                                  #print(a.reshape(-1,2)) #-1就是不管行数，控制列数  output:3*2矩阵
                                  #b = a.reshape(2,3)
                                  # print(a)              output:[1,2,3,4,5,6]
                                  # print(b)              output:2*3矩阵
    h = relu(nd.dot(X,w1)+b1)
    return  nd.dot(h,w2)+b2

#损失函数
loss = gloss.SoftmaxCrossEntropyLoss()

batch_size = 256
train_iter,test_iter = my.load_data_fashion_mnist(batch_size)
num_input,num_output,num_hiddens = 784,10,256
w1 = nd.random.normal(scale=0.01,shape=(num_input,num_hiddens))
b1 = nd.zeros(num_hiddens)
w2 = nd.random.normal(scale=0.01,shape=(num_hiddens,num_output))
b2 = nd.zeros(num_output)
params = [w1,b1,w2,b2]
for param in params:
    param.attach_grad()

num_epochs = 5
lr = 0.1
my.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,params,lr)


for X, y in test_iter:
    break
true_labels = my.get_fashion_mnist_labels(y.asnumpy())
pred_labels = my.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
my.show_fashion_mnist(X[0:9], titles[0:9])

