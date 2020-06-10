""" 线性回归"""
from IPython import display
import matplotlib.pyplot as plt
from mxnet import  autograd,nd
import  random

##生成数据集
def Generate_data(num_inputs,num_examples):
    true_w = [2,-3.4]
    true_b = 4.2
    features = nd.random.normal(scale=1,shape=(num_examples,num_inputs))
    labels = true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b
    labels+=nd.random.normal(scale=0.01,shape=labels.shape)
    ####其中features是训练数据特征，labels是标签(y=Xw+b+噪声 噪声服从均值为0 标准差为0.01的正态分布)。
    return features,labels
########################################
def use_svg_display():#用矢量图显示
    display.set_matplotlib_formats("svg")
def set_figsize(figsize=(3.5,2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize']=figsize

###读取数据集
def batch_iter(batch_size,features,labels):
    num_examples = len(features)
    indices=list(range(num_examples))
    random.shuffle(indices)#随机读取样本
    for i in range(0,num_examples,batch_size):
        j=nd.array(indices[i:min(i+batch_size,num_examples)])
        yield features.take(j),labels.take(j) #take函数是根据索引返回对应的元素


#线性模型
def Linreg(X,w,b):
    return nd.dot(X,w)+b

#定义损失函数
def squared_loss(y_hat,y):
    return (y_hat - y.reshape(y_hat.shape))**2/2#shape读取矩阵长度 reshape返回一个矩阵

#定义优化算法
def sgd(params,lr,batch_size):
    for param in params:
        param[:] = param-lr*param.grad/batch_size
if __name__ == '__main__':
    num_inputs = 2
    num_examples = 1000

    features, labels = Generate_data(num_inputs,num_examples)
    set_figsize()
    plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1)
    plt.show()
    batch_size=10

    w=nd.random.normal(scale=0.01,shape=(num_inputs,1))#定义权重 偏置
    b=nd.zeros(shape=(1,))

    w.attach_grad()
    b.attach_grad() #为了求有关变量x的梯度，我们需要先调用attach_grad函数来申请存储梯度所需要的内存。

    lr=0.03
    num_epochs=3
    net = Linreg
    loss = squared_loss
    for epoch in range(num_epochs):
        for X,y in batch_iter(batch_size,features,labels):
            with autograd.record():
                l=loss(net(X,w,b),y)
            l.backward()
            sgd([w,b],lr,batch_size)
        train_l=loss(net(features,w,b),labels)
        print("epoch %d, loss  %f" % (epoch+1,train_l.mean().asnumpy()))