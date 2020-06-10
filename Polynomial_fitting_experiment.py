"""多项式函数拟合实验"""
import Myself_d2lzh as my
from mxnet import nd,autograd,gluon
from mxnet.gluon import data as gdata, loss as gloss, nn
from matplotlib import pyplot as plt
###########################################################
###说明#####
# ======================定义模型
"""from mxnet.gluon import nn
# 导入nn模块。实际上，“nn”是 neural networks（神经网络）的缩写。
# 顾名思义，该模块定义了大量神经网络的层。我们先定义一个模型变量net，它是一个 Sequential 实例。
# 在 Gluon 中，Sequential 实例可以看作是一个串联各个层的容器。
# 在构造模型时，我们在该容器中依次添加层。当给定输入数据时，容器中的每一层将依次计算并将输出作为下一层的输入。
net = nn.Sequential()  # 这一开始是空的，定义的容器
# 线性模型其实就是单层网络，单层网络的输入取决于输出，输出一个y
# 作为一个单层神经网络，线性回归输出层中的神经元和输入层中各个输入完全连接。
# 因此，线性回归的输出层又叫全连接层。在 Gluon 中，全连接层是一个Dense实例。我们定义该层输出个数为 1。
net.add(nn.Dense(1))
# 在 Gluon 中我们无需指定每一层输入的形状，例如线性回归的输入个数
# 当模型看见数据时，例如后面执行net(X)时，模型将自动推断出每一层的输入个数。

# =========================初始化模型参数
# 在使用net前，我们需要初始化模型参数，例如线性回归模型中的权重和偏差。
# 我们从 MXNet 导入initializer模块。该模块提供了模型参数初始化的各种方法。
# 这里的init是initializer的缩写形式。我们通过init.Normal(sigma=0.01)指定权重
# 参数每个元素将在初始化时随机采样于均值为 0 标准差为 0.01 的正态分布。偏差参数默认会初始化为零。
from mxnet import init
net.initialize(init.Normal(sigma=0.01))
# ====================定义损失函数
# 在 Gluon 中，loss模块定义了各种损失函数。
# 我们用假名gloss代替导入的loss模块，并直接使用它所提供的平方损失作为模型的损失函数
from mxnet.gluon import loss as gloss

loss = gloss.L2Loss()  # 平方损失又称L2范数损失

# ============定义优化函数
# 同样，我们也无需实现小批量随机梯度下降。在导入 Gluon 后，我们创建一个Trainer实例，
# 并指定学习率为 0.03 的小批量随机梯度下降（sgd）为优化算法。该优化算法将用来迭代net实例
# 所有通过add函数嵌套的层所包含的全部参数。这些参数可以通过collect_params函数获取。
from mxnet import gluon

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

# ===============训练模型
# 在使用 Gluon 训练模型时，我们通过调用Trainer实例的step函数来迭代模型参数。
# 上一博文中提到，由于变量l是长度为batch_size的一维 NDArray，(L小写l)
# 执行l.backward()等价于执行l.sum().backward()。按照小批量随机梯度下降的定义，
# 我们在step函数中指明批量大小，从而对批量中样本梯度求平均。
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch %d,loss: %f' % (epoch, l.mean().asnumpy()))

————————————————
版权声明：本文为CSDN博主「Mangoit」的原创文章，遵循
CC
4.0
BY - SA
版权协议，转载请附上原文出处链接及本声明。
原文链接：https: // blog.csdn.net / qq_36666756 / article / details / 83186349 """


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    """Plot x and log(y)."""
    my.set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals,y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()

def fit_and_plot(train_features,test_features,train_labels,test_labels,):
    print("train_features.len %d" % (len(train_features)))
    print("train_features.shape" ,(train_features.shape))

    print("test_features.len %d" % (len(test_features)))
    print("test_features.shape", (test_features.shape))

    net = nn.Sequential()    #gluon.nn：神经网络;Sequential是个类。Dense：全连接。
    net.add(nn.Dense(1))    #Dense(1)：表示输出值的维度（一层的神经网络相当于线性回归）
    net.initialize()        ## 参数初始化
    batch_size = min(10,train_labels.shape[0])
    train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features,train_labels),
                                  batch_size,shuffle=True)
    trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate' : 0.01})
                                #trainer 训练器；仅保存参数及超参，以及根据 batch size 进行参数更新
    train_ls,test_ls = [],[]
    num_epochs,loss = 100,gloss.L2Loss()
    for _ in range(num_epochs):
        for X,y in train_iter:
            with autograd.record():
                l=loss(net(X),y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(loss(net(train_features), train_labels).mean().asscalar())  # 将训练损失保存到train_ls中
        test_ls.append(loss(net(test_features), test_labels).mean().asscalar())  # 将测试损失保存到test_ls中
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1,num_epochs+1),train_ls,'epochs','loss',range(1,num_epochs+1),
                 test_ls,['train','test'])
    print('weight:', net[0].weight.data().asnumpy(),
          '\nbias:', net[0].bias.data().asnumpy())


if __name__ == '__main__':
    n_train,n_test,true_w,true_b = 100,100,[1.2,-3.4,5.6],5
    features = nd.random.normal(shape=(n_train+n_test,1))
    poly_features = nd.concat(features,nd.power(features,2),nd.power(features,3)) #concat拼接矩阵                                                                                 #nd.power幂运算
    print(poly_features.shape)
    print(len(poly_features))
    labels = (true_w[0]*poly_features[:,0]+true_w[1]*poly_features[:,1]
              +true_w[2]*poly_features[:,2]+true_b)
    labels += nd.random.normal(scale=0.1,shape=labels.shape)
    print(features[:2])
    print(poly_features[:2])
    print(labels[:2])

    """三阶多项式函数拟合：
    train_features:100*3  [features,features**2,features**3]  表示100个数据，一个数据3个值 也就是一个X为1*3
    test_features:100*3
    模型为 y=Xw+b,输出为1个标量  故w为3*1，b为标量  
    正常  
    output:final epoch: train loss 0.006970712 test loss 0.006331955
           weight: [[ 1.1752776 -3.391078   5.60141  ]] 
           bias: [4.9874315]
    """
    # fit_and_plot(poly_features[:n_train, :],poly_features[n_train:, :],
    #            labels[:n_train],labels[n_train:])




    """线性函数拟合  
    train_features:100*1 表示100个数据，一个数据一个数  也就是一个X为1*1
    test_features:100*1
    y=Xw+b 也就是w:1*1(标量) b：标量 
    这个模型就是一个线性函数，
    欠拟合 
    
    output:final epoch: train loss 159.33392 test loss 103.20967
           weight: [[22.693125]] 
           bias: [-0.67537934]
    loss很大，欠拟合：模型无法得到较低的训练误差
    """
    #fit_and_plot(features[:n_train, :],features[n_train:, :],
    #             labels[:n_train],labels[n_train:])


    """训练样本不足
    train_feature:2*3 表示两组数据
    test_features:100*3
    模型为 y=Xw+b,输出为1个标量  故w为3*1，b为标量
    output:  final epoch: train loss 0.48284656 test loss 134.29808
             weight: [[1.9822394 1.9652436 2.0623446]] 
             bias: [2.498562] 
    过拟合
    训练数据严重不足，而模型的参数（w中就是3个，一个b）多，显得模型复杂。在训练误差显得很低，但是在测试数据集的误差很大这就是典型的过拟合
    
    """
    fit_and_plot(poly_features[0:2, :],poly_features[n_train:, :],
               labels[0:2],labels[n_train:])

