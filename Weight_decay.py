"""权重衰减
虽然增大训练数据集可能会减轻过拟合，
但是获取额外的训练数据往往代价高昂。
应对过拟合问题的常用方法：权重衰减（weight decay


权重衰减等价于L2范数正则化（regularization）。
正则化通过为模型损失函数添加惩罚项使学出的模型参数值较小，是应对过拟合的常用手段。

L2范数正则化在模型原损失函数基础上添加L2范数惩罚项
L2范数惩罚项指的是模型权重参数每个元素的平方和与一个正的常数的乘积
新的损失函数： ℓ(w1,w2,b)+（λ/2n）*（||w||**2）,λ>0 超参数   n为样本数

"""
###高维线性实验###
import Myself_d2lzh as my
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn

##获取数据集##
n_train,n_test,num_inputs=20,100,200
true_w= nd.ones(shape=(num_inputs,1))*0.01
true_b =0.05
features = nd.random.normal(shape=(n_train+n_test,num_inputs))
labels = nd.dot(features,true_w)+true_b
labels +=nd.random.normal(scale=0.01,shape=labels.shape)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]

batch_size=1
num_epochs= 100
learning_rate= 0.003
net = my.linreg
loss = my.squared_loss
train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features,train_labels),batch_size,shuffle=True)


def fit_and_plot(lanbd):
    w,b=init_params()
    train_ls,test_ls = [],[]
    for _ in range(num_epochs):
        for X,y in train_iter:
            with autograd.record():
                l=loss(net(X,w,b),y)+lanbd*l2_penalty(w)
            l.backward()
            my.sgd([w,b],learning_rate,batch_size)
        train_ls.append(loss(net(train_features,w,b),train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().asscalar())
    my.semilogy(lanbd,range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('𝜆= %d' % (lanbd),'   L2 norm of w:', w.norm().asscalar())
    #初始化参数#
def init_params():
    w = nd.random.normal(scale=1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    w.attach_grad()
    b.attach_grad()
    return [w, b]

##L2 范数惩罚项##
def l2_penalty(w):
    return (w**2).sum() / 2

fit_and_plot(0) #λ=lanbd=0 表示没用使用权重衰减
""" output:L2 norm of w: 13.155677 """

fit_and_plot(5) #λ=lanbd=3 表示使用权重衰减
"""output: L2 norm of w: 0.042180628"""
fit_and_plot(15)
fit_and_plot(20)
fit_and_plot(50)

