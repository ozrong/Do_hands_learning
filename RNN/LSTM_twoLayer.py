import math
import time
from mxnet.gluon import loss as gloss
from mxnet import nd,autograd
import random
import zipfile
import d2lzh as d2l

def load_data_jay_lyrics():
    #打开数据集，预处理以下
    with zipfile.ZipFile("E:\CODE\python_code\Do_hands_learning\data\jaychou_lyrics.txt.zip") as zin:
        with zin.open("jaychou_lyrics.txt") as f:
            corpus_chars = f.read().decode("utf-8")
    corpus_chars[:40]
    corpus_chars=corpus_chars.replace("\n",",").replace("\r"," ")
    corpus_chars=corpus_chars[:10000]

    idx_to_char = list(set(corpus_chars))#是文字
    #print(idx_to_char)
    char_to_idx = dict([(char,i) for i,char in enumerate(idx_to_char)])#字典 例如 ‘害’：0
    # print(char_to_idx['害'])
    # print(char_to_idx)
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size
def to_onehot(X,size):
    input=[]
    for x in X.T:
        temp=nd.one_hot(x, size)
        input.append(temp)
    # input=[nd.one_hot(x,size) for x in X.T]
    return input #返回的是一个矩阵 一列就是一个one_hot

"""我们初始化模型参数。隐藏单元个数 num_hiddens是一个超参数"""
def get_params():
    def _one(shape):
        return nd.random.normal(scale=0.01, shape=shape, ctx=ctx)
    def _three():
        return (_one((num_inputs,num_hiddens)),
                _one((num_hiddens,num_hiddens)),
                nd.zeros(num_hiddens,ctx=ctx))
    #第一层参数
    W_xi,W_hi,b_i = _three()#输入门参数
    W_xf,W_hf,b_f = _three()#遗忘门参数
    W_xo,W_ho,b_o = _three()#输出门参数
    W_xc,W_hc,b_c = _three()#候选记忆细胞参数
    #第二层参数
    W_xi2,W_hi2,b_i2 = _three()#输入门参数
    W_xi2=_one((num_hiddens,num_hiddens))
    W_xf2,W_hf2,b_f2 = _three()#遗忘门参数
    W_xf2 = _one((num_hiddens, num_hiddens))
    W_xo2,W_ho2,b_o2 = _three()#输出门参数
    W_xo2 = _one((num_hiddens, num_hiddens))
    W_xc2,W_hc2,b_c2 = _three()#候选记忆细胞参数
    W_xc2 = _one((num_hiddens, num_hiddens))


    #输出层参数
    W_hq=_one((num_hiddens,num_outputs))
    b_q=nd.zeros(num_outputs,ctx=ctx)

    params = [ W_xi,W_hi,b_i,W_xf,W_hf,b_f,W_xo,W_ho,b_o,W_xc,W_hc,b_c ,
               W_xi2,W_hi2,b_i2,W_xf2,W_hf2,b_f2,W_xo2,W_ho2,b_o2, W_xc2,W_hc2,b_c2,
               W_hq,b_q]
    for param in params:
        param.attach_grad()
    return params

"""定义模型"""
def init_rnn_state(batch_size, num_hiddens, ctx):
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx),#初始化隐藏状态
            nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx) )#初始化记忆细胞

"""rnn函数定义了在一个时间步里如何计算隐藏状态和输出。
这里的激活函数使用了tanh函数"""
def lstm(inputs, state, params):
    # inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
    [ W_xi,W_hi,b_i,W_xf,W_hf,b_f,W_xo,W_ho,b_o,W_xc,W_hc,b_c ,
      W_xi2,W_hi2,b_i2,W_xf2,W_hf2,b_f2,W_xo2,W_ho2,b_o2, W_xc2,W_hc2,b_c2,
      W_hq,b_q] = params
    (H,C) = state
    outputs = []
    for X in inputs:
        I = nd.sigmoid(nd.dot(X,W_xi)+nd.dot(H,W_hi)+b_i)
        F = nd.sigmoid(nd.dot(X,W_xf)+nd.dot(H,W_hf)+b_f)
        O = nd.sigmoid(nd.dot(X,W_xo)+nd.dot(H,W_ho)+b_o)
        C_tilda = nd.tanh(nd.dot(X,W_xc)+nd.dot(H,W_hc)+b_c)
        C1=F*C+I*C_tilda
        H1=C.tanh()*O

        I2 = nd.sigmoid(nd.dot(H,W_xi2)+nd.dot(H1,W_hi2)+b_i2)
        F2 = nd.sigmoid(nd.dot(H,W_xf2)+nd.dot(H1,W_hf2)+b_f2)
        O2 = nd.sigmoid(nd.dot(H,W_xo2)+nd.dot(H1,W_ho2)+b_o)
        C_tilda = nd.tanh(nd.dot(H,W_xc2)+nd.dot(H1,W_hc2)+b_c2)
        C2=F2*C1+I2*C_tilda
        H2=C2.tanh()*O2


        Y=nd.dot(H2,W_hq)+b_q
        outputs.append(Y)
    return outputs, (H2,C2)


"""定义预测函数
以下函数基于前缀prefix（含有数个字符的字符串）来预测接下来的num_chars个字符。
这个函数稍显复杂，其中我们将循环神经单元rnn设置成了函数参数，
这样在后面小节介绍其他循环神经网络时能重复使用这个函数
"""
# 本函数已保存在d2lzh包中方便以后使用
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, ctx)
    output = [char_to_idx[prefix[0]]]

    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        a=output[-1]
        X = to_onehot(nd.array([a], ctx=ctx), vocab_size)
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)
        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])

"""循环神经网络中较容易出现梯度衰减或梯度爆炸,为了应对梯度爆炸，我们可以裁剪梯度"""
# 本函数已保存在d2lzh包中方便以后使用
def grad_clipping(params, theta, ctx):
    norm = nd.array([0], ctx)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm



"""定义模型训练函数"""
# 本函数已保存在d2lzh包中方便以后使用
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, ctx, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive
    params = get_params()
    loss = gloss.SoftmaxCrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:  # 如使用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, ctx)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx)
        for X, Y in data_iter:
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, ctx)
            else:  # 否则需要使用detach函数从计算图分离隐藏状态
                for s in state:
                    s.detach()
            with autograd.record():
                inputs = to_onehot(X, vocab_size)
                # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
                (outputs, state) = rnn(inputs, state, params)
                # 连结之后形状为(num_steps * batch_size, vocab_size)
                outputs = nd.concat(*outputs, dim=0)
                # Y的形状是(batch_size, num_steps)，转置后再变成长度为
                # batch * num_steps 的向量，这样跟输出的行一一对应
                y = Y.T.reshape((-1,))
                # 使用交叉熵损失计算平均分类误差
                l = loss(outputs, y).mean()
            l.backward()
            grad_clipping(params, clipping_theta, ctx)  # 裁剪梯度
            d2l.sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不用再做平均
            l_sum += l.asscalar() * y.size
            n += y.size

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(
                    prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx))




"""随机采样"""
# 本函数已保存在d2lzh包中方便以后使用
def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    # 减1是因为输出的索引是相应输入的索引加1
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # 返回从pos开始的长为num_steps的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield nd.array(X, ctx), nd.array(Y, ctx)
if __name__ == '__main__':
    (corpus_indices,char_to_idx,idx_to_char,vocab_size)=load_data_jay_lyrics()
    num_inputs,num_hiddens,num_outputs = vocab_size,256,vocab_size
    ctx = d2l.try_gpu();
    print("will use",ctx)
    """测试onehot"""
    """    test = nd.arange(10).reshape(2,5)
    print(test)
    inputs=to_onehot(test,vocab_size)
    print(len(inputs),inputs[0].shape)
    print(inputs[0])"""


    # params = get_params()
    # per=predict_rnn('分开', 10, rnn, params, init_rnn_state, num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx)
    # print(per)



    num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']



    train_and_predict_rnn(lstm, get_params, init_rnn_state, num_hiddens,
                      vocab_size, ctx, corpus_indices, idx_to_char,
                      char_to_idx, False, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)





