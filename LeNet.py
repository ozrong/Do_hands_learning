
import d2lzh as d2l
import Myself_d2lzh as my
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
import time

net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, activation='sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        # Dense会默认将(批量大小, 通道, 高, 宽)形状的输入转换成
        # (批量大小, 通道 * 高 * 宽)形状的输入
        nn.Dense(120, activation='sigmoid'),
        nn.Dense(84, activation='sigmoid'),
        nn.Dense(10))

# X = nd.random.uniform(shape=(1, 1, 28, 28))
# net.initialize()
# for layer in net:
#     X = layer(X)
#     print(layer.name, 'output shape:\t', X.shape)
"""
    conv0 output shape:	 (1, 6, 24, 24)
    pool0 output shape:	 (1, 6, 12, 12)
    conv1 output shape:	 (1, 16, 8, 8)
    pool1 output shape:	 (1, 16, 4, 4)
    dense0 output shape:	 (1, 120)
    dense1 output shape:	 (1, 84)
    dense2 output shape:	 (1, 10)
"""
    



"""检测是否可以使用GPU"""

def try_gpu():  # 本函数已保存在d2lzh包中方便以后使用
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx
ctx = try_gpu()

"""准确率"""
def evaluate_accuracy(data_iter, net, ctx):
    acc_sum, n = nd.array([0], ctx=ctx), 0
    for X, y in data_iter:
        # 如果ctx代表GPU及相应的显存，将数据复制到显存上
        X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum()
        n += y.size
    return acc_sum.asscalar() / n

"""训练模型"""
def train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx,
              num_epochs):
    print('training on', ctx)
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc,
                 time.time() - start))




"""获取数据集"""
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)


lr, num_epochs = 0.9,5
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)

for X, y in test_iter:
    break
true_labels = my.get_fashion_mnist_labels(y.asnumpy())
pred_labels = my.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
my.show_fashion_mnist(X[0:12], titles[0:12])
"""
epoch=5 lr=0.9
epoch 1, loss 2.3143, train acc 0.109, test acc 0.259, time 32.5 sec
epoch 2, loss 1.3171, train acc 0.480, test acc 0.634, time 34.3 sec
epoch 3, loss 0.8533, train acc 0.668, test acc 0.697, time 34.3 sec
epoch 4, loss 0.7096, train acc 0.721, test acc 0.746, time 33.5 sec
epoch 5, loss 0.6347, train acc 0.749, test acc 0.771, time 31.6 sec


epoch=7 lr=0.9
epoch 1, loss 2.3168, train acc 0.109, test acc 0.280, time 31.2 sec
epoch 2, loss 1.3466, train acc 0.476, test acc 0.679, time 31.4 sec
epoch 3, loss 0.8503, train acc 0.671, test acc 0.723, time 31.6 sec
epoch 4, loss 0.7140, train acc 0.721, test acc 0.740, time 32.5 sec
epoch 5, loss 0.6422, train acc 0.746, test acc 0.775, time 32.1 sec
epoch 6, loss 0.5785, train acc 0.771, test acc 0.792, time 31.6 sec
epoch 7, loss 0.5432, train acc 0.787, test acc 0.813, time 32.4 sec



epoch=10 lr=0.9
epoch 1, loss 2.3156, train acc 0.106, test acc 0.119, time 30.4 sec
epoch 2, loss 1.3333, train acc 0.477, test acc 0.642, time 29.4 sec
epoch 3, loss 0.8516, train acc 0.668, test acc 0.709, time 34.4 sec
epoch 4, loss 0.7124, train acc 0.721, test acc 0.739, time 36.3 sec
epoch 5, loss 0.6385, train acc 0.746, test acc 0.767, time 33.6 sec
epoch 6, loss 0.5786, train acc 0.772, test acc 0.785, time 31.2 sec
epoch 7, loss 0.5388, train acc 0.789, test acc 0.814, time 30.0 sec
epoch 8, loss 0.5065, train acc 0.804, test acc 0.814, time 29.8 sec
epoch 9, loss 0.4758, train acc 0.817, test acc 0.828, time 29.2 sec
epoch 10, loss 0.4557, train acc 0.828, test acc 0.842, time 29.3 sec

epoch = 20 lr=0.5
epoch 1, loss 2.3191, train acc 0.099, test acc 0.100, time 36.7 sec
epoch 2, loss 1.8521, train acc 0.308, test acc 0.559, time 34.5 sec
epoch 3, loss 1.0025, train acc 0.606, test acc 0.672, time 33.2 sec
epoch 4, loss 0.8351, train acc 0.682, test acc 0.714, time 33.4 sec
epoch 5, loss 0.7414, train acc 0.711, test acc 0.734, time 32.1 sec
epoch 6, loss 0.6864, train acc 0.728, test acc 0.749, time 32.2 sec
epoch 7, loss 0.6392, train acc 0.746, test acc 0.763, time 32.5 sec
epoch 8, loss 0.6062, train acc 0.760, test acc 0.768, time 32.5 sec
epoch 9, loss 0.5741, train acc 0.775, test acc 0.789, time 34.2 sec
epoch 10, loss 0.5450, train acc 0.787, test acc 0.801, time 32.8 sec
epoch 11, loss 0.5222, train acc 0.796, test acc 0.801, time 32.4 sec
epoch 12, loss 0.5022, train acc 0.806, test acc 0.820, time 32.1 sec
epoch 13, loss 0.4805, train acc 0.819, test acc 0.828, time 32.1 sec
epoch 14, loss 0.4652, train acc 0.825, test acc 0.833, time 32.4 sec
epoch 15, loss 0.4522, train acc 0.832, test acc 0.845, time 32.3 sec
epoch 16, loss 0.4377, train acc 0.839, test acc 0.850, time 32.1 sec
epoch 17, loss 0.4257, train acc 0.843, test acc 0.840, time 33.2 sec
epoch 18, loss 0.4150, train acc 0.848, test acc 0.850, time 32.6 sec
epoch 19, loss 0.4054, train acc 0.852, test acc 0.859, time 35.3 sec
epoch 20, loss 0.3939, train acc 0.856, test acc 0.861, time 35.9 sec




epoch=10 lr=0.5 datasets=mnist
epoch 1, loss 2.3195, train acc 0.103, test acc 0.103, time 30.1 sec
epoch 2, loss 2.3054, train acc 0.111, test acc 0.103, time 33.9 sec
epoch 3, loss 2.2998, train acc 0.116, test acc 0.114, time 35.6 sec
epoch 4, loss 1.9048, train acc 0.338, test acc 0.736, time 34.5 sec
epoch 5, loss 0.5774, train acc 0.830, test acc 0.898, time 36.4 sec
epoch 6, loss 0.3035, train acc 0.910, test acc 0.928, time 35.0 sec
epoch 7, loss 0.2205, train acc 0.934, test acc 0.943, time 32.2 sec
epoch 8, loss 0.1750, train acc 0.948, test acc 0.958, time 30.0 sec
epoch 9, loss 0.1469, train acc 0.956, test acc 0.964, time 30.8 sec
epoch 10, loss 0.1270, train acc 0.962, test acc 0.970, time 31.3 sec
"""