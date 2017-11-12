import mxnet as mx
import numpy as np
# data.shape = [batch_size, 1, 28, 28]
data = mx.sym.Variable('data')

conv1_num_filter = 256
kernel = (9, 9)
batch_size = 1
epsilon = 1e-08
input_shape =(1, 28, 28)
#Conv2D layer
# net.shape = [batch_size, 256, 20, 20]
#net = mx.sym.Reshape(data=data, shape=(0., -4, -1, 1, 0, 0))
#print('after reshape', net.infer_shape(data=(batch_size, 28, 28)))
net = mx.sym.Convolution(data=data,
                         num_filter=conv1_num_filter,
                         kernel=kernel,
                         layout='NCHW',
                         name='conv1')
net = mx.sym.Activation(data=net, act_type='relu', name='conv1_act')
print('after conv2d',net.infer_shape(data=(batch_size, 1, 28, 28)))
# net.shape = [batch_size, 256, 6, 6]
dim_vector = 8
n_channels = 32
squash_axis = 2
net = mx.sym.Convolution(data=net,
                         num_filter=dim_vector*n_channels,
                         kernel=kernel,
                         stride=[2, 2],
                         name='conv2_primarycap'
                         )
print('after conv2d 2',net.infer_shape(data=(batch_size, 1, 28, 28)))
# net.shape = [batch_size, 1152,8]
net = mx.sym.Reshape(data=net, shape=(0, -1, dim_vector))


def squash(net, name=''):
    s_squared_norm = mx.sym.sum(data=mx.sym.square(net, name='square_'+name), axis=squash_axis, keepdims=True, name='s_squared_norm_'+name)
    scale = s_squared_norm / (1 + s_squared_norm) / mx.sym.sqrt(data=(s_squared_norm+epsilon), name='s_squared_norm_sqrt_'+name)
    squashed_net = mx.sym.broadcast_mul(scale, net, name='squashed_net_'+name)
    return squashed_net

net = squash(net)
net.infer_shape(data=(batch_size, 1, 28, 28))
print('after squash', net.infer_shape(data=(batch_size, 1, 28, 28)))
# CapsuleLayer

# int_num_capsule = n_class
# input_shape = [None, input_num_capsule
# ('primarycaps', TensorShape([Dimension(None), Dimension(1152), Dimension(8)]))
# ('W', TensorShape([Dimension(1152), Dimension(10), Dimension(8), Dimension(16)]))
# ('B', TensorShape([Dimension(1), Dimension(1152), Dimension(10), Dimension(1), Dimension(1)]))

# ('inputs_hat', TensorShape([Dimension(None), Dimension(1152), Dimension(10), Dimension(1), Dimension(16)]))
# ('i', 0, 'c', TensorShape([Dimension(1), Dimension(1152), Dimension(10), Dimension(1), Dimension(1)]))
# ('i', 0, 'outputs', TensorShape([Dimension(None), Dimension(1), Dimension(10), Dimension(1), Dimension(16)]))
# ('bias', TensorShape([Dimension(None), Dimension(1152), Dimension(10), Dimension(1), Dimension(1)]))
# ('i', 1, 'c', TensorShape([Dimension(None), Dimension(1152), Dimension(10), Dimension(1), Dimension(1)]))
# ('i', 1, 'outputs', TensorShape([Dimension(None), Dimension(1), Dimension(10), Dimension(1), Dimension(16)]))
# ('bias', TensorShape([Dimension(None), Dimension(1152), Dimension(10), Dimension(1), Dimension(1)]))
# ('i', 2, 'c', TensorShape([Dimension(None), Dimension(1152), Dimension(10), Dimension(1), Dimension(1)]))
# ('i', 2, 'outputs', TensorShape([Dimension(None), Dimension(1), Dimension(10), Dimension(1), Dimension(16)]))

input_num_capsule = 1152
input_dim_vector = 8
n_class = 10
num_capsule = n_class
dim_vector = 16
num_routing = 3
kernel_initializer = mx.init.Xavier(rnd_type='uniform', factor_type='avg', magnitude=3)
bias_initializer = mx.init.Zero()

# build w and bias
# ('W', TensorShape([Dimension(1152), Dimension(10), Dimension(8), Dimension(16)]))
# ('B', TensorShape([Dimension(1), Dimension(1152), Dimension(10), Dimension(1), Dimension(1)]))
w = mx.sym.Variable('Weight',
                    shape=(1, input_num_capsule, num_capsule, input_dim_vector, dim_vector),
                    init=kernel_initializer)
bias = mx.sym.Variable('Bias',
                       shape=(batch_size, input_num_capsule, num_capsule, 1, 1),
                       init=bias_initializer)

# ('input', TensorShape([Dimension(None), Dimension(1152), Dimension(8)]))
# ('inputs_expand', TensorShape([Dimension(None), Dimension(1152), Dimension(1), Dimension(8), Dimension(1)]))
net = mx.sym.Reshape(data=net, shape=(0, 0, -4, -1, 1))
net = mx.sym.Reshape(data=net, shape=(0, 0, -4, 1, -1, 0))
print('after inputs_expand', net.infer_shape(data=(batch_size, 1, 28, 28)))
# input_tiled
# ('inputs_tiled.shape', TensorShape([Dimension(None), Dimension(1152), Dimension(10), Dimension(8), Dimension(1)]))
net = mx.sym.tile(data=net, reps=(1, 1, n_class, 1, 1))
net.infer_shape(data=(batch_size, 1, 28, 28))
# w_tiled
# w_tiled.shape = [(1L, 1152L, 10L, 8L, 16L)]
w_tiled = mx.sym.tile(w, reps=(batch_size, 1, 1, 1, 1))
w_tiled.infer_shape()
inputs_hat = mx.sym.linalg_gemm2(w_tiled, net, transpose_a=True)
inputs_hat = mx.sym.swapaxes(data=inputs_hat, dim1=3, dim2=4)
inputs_hat.infer_shape(data=(batch_size, 1, 28, 28))
# ('i', 0, 'c', TensorShape([Dimension(1), Dimension(1152), Dimension(10), Dimension(1), Dimension(1)]))

for i in range(0, num_routing):
    c = mx.sym.softmax(bias, axis=2, name='c'+str(i))
    #print('i',i,' c',c.infer_shape())
    outputs = squash(mx.sym.sum(mx.sym.broadcast_mul(c, inputs_hat, name='broadcast_mul_'+str(i)), axis=1, keepdims=True, name='sum_'+str(i)), name='output_'+str(i))
    #print('i', i, ' output', outputs.infer_shape(data=(1, 28, 28)))
    if i != num_routing - 1:
        bias += mx.sym.sum(mx.sym.broadcast_mul(c, inputs_hat, name='bias_broadcast_mul'+str(i)), axis=4, keepdims=True, name='bias_'+str(i))

# ('digitcaps', TensorShape([Dimension(None), Dimension(10), Dimension(16)]))
digitcaps = mx.sym.Reshape(data=outputs, shape=(-1, num_capsule, dim_vector))
digitcaps.infer_shape(data=(batch_size, 1, 28, 28))
# ('i', 0, 'outputs', TensorShape([Dimension(None), Dimension(1), Dimension(10), Dimension(1), Dimension(16)]))
# ('bias', TensorShape([Dimension(None), Dimension(1152), Dimension(10), Dimension(1), Dimension(1)]))
# ('i', 1, 'c', TensorShape([Dimension(None), Dimension(1152), Dimension(10), Dimension(1), Dimension(1)]))
# ('i', 1, 'outputs', TensorShape([Dimension(None), Dimension(1), Dimension(10), Dimension(1), Dimension(16)]))
# ('bias', TensorShape([Dimension(None), Dimension(1152), Dimension(10), Dimension(1), Dimension(1)]))
# ('i', 2, 'c', TensorShape([Dimension(None), Dimension(1152), Dimension(10), Dimension(1), Dimension(1)]))
# ('i', 2, 'outputs', TensorShape([Dimension(None), Dimension(1), Dimension(10), Dimension(1), Dimension(16)]))

# ('out_caps', TensorShape([Dimension(None), Dimension(10)]))
# ('inputs_masked', TensorShape([Dimension(None), Dimension(16)]))

out_caps = mx.sym.sqrt(data=mx.sym.sum(digitcaps, 2))
out_caps.infer_shape(data=(batch_size, 1, 28, 28))
y = mx.sym.Variable('softmax_label', shape=(batch_size))
y = mx.sym.one_hot(y, n_class)
y = mx.sym.Reshape(data=y, shape=(batch_size, -4, n_class, -1))
y.infer_shape(softmax_label=(batch_size))
#inputs_masked = mx.sym.batch_dot(mx.sym.transpose(y), digitcaps)
inputs_masked = mx.sym.linalg_gemm2(y, digitcaps, transpose_a=True)
#inputs_masked = mx.sym.linalg_gemm(mx.sym.transpose(y), digitcaps, transpose_a=True)
# y.infer_shape(y=(1,10))
# inputs_masked = mx.sym.linalg_gemm(out_caps, y, transpose_a=True)
#inputs_masked = mx.sym.batch_dot(out_caps, y)
print(inputs_masked.infer_shape(data=(batch_size, 1, 28, 28), softmax_label=(batch_size)))
inputs_masked = mx.sym.Reshape(data=inputs_masked, shape=(-3, 0))
print(inputs_masked.infer_shape(data=(batch_size, 1, 28, 28), softmax_label=(batch_size)))
x_recon = mx.sym.FullyConnected(data=inputs_masked, num_hidden=512, name='x_recon')
x_recon = mx.sym.Activation(data=x_recon, act_type='relu', name='x_recon_act')
x_recon = mx.sym.FullyConnected(data=x_recon, num_hidden=1024, name='x_recon2')
x_recon = mx.sym.Activation(data=x_recon, act_type='relu', name='x_recon_act2')

x_recon = mx.sym.FullyConnected(data=x_recon, num_hidden=np.prod(input_shape), name='x_recon3')
x_recon = mx.sym.Activation(data=x_recon, act_type='relu', name='x_recon_act3')
print(x_recon.infer_shape(data=(batch_size, 1, 28, 28), softmax_label=(batch_size)))
def margin_loss(y_true, y_pred):
    L = y_true * mx.sym.square(mx.sym.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * mx.sym.square(mx.sym.maximum(0., y_pred - 0.1))
    return mx.sym.mean(data=mx.sym.sum(L, 1))

data_flatten = mx.sym.flatten(data=data)
loss = mx.symbol.MakeLoss(margin_loss(data_flatten, x_recon))
out_caps_blocked = out_caps
out_caps_blocked = mx.sym.BlockGrad(out_caps_blocked)
final_net = mx.sym.Group([out_caps_blocked, loss])
# get mnist data set

import numpy as np
import os
import urllib
import gzip
import struct
def download_data(url, force_download=True):
    fname = url.split("/")[-1]
    if force_download or not os.path.exists(fname):
        urllib.urlretrieve(url, fname)
    return fname

def read_data(label_url, image_url):
    with gzip.open(download_data(label_url)) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(download_data(image_url), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)

path='http://yann.lecun.com/exdb/mnist/'
(train_lbl, train_img) = read_data(
    path+'train-labels-idx1-ubyte.gz', path+'train-images-idx3-ubyte.gz')
(val_lbl, val_img) = read_data(
    path+'t10k-labels-idx1-ubyte.gz', path+'t10k-images-idx3-ubyte.gz')

def to4d(img):
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255

train_iter = mx.io.NDArrayIter(to4d(train_img), train_lbl, batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(to4d(val_img), val_lbl, batch_size)

class LossMetric(mx.metric.EvalMetric):
    def __init__(self, batch_size, num_gpu, is_epoch_end=False, is_logging=True):
        super(LossMetric, self).__init__('STTMetric')
        self.batch_size = batch_size
        self.num_gpu = num_gpu
    def update(self, labels, preds):
        self.batch_loss = 0.
        for label, pred in zip(labels[0],preds[0]):
            print(label.asnumpy())
            print(pred.asnumpy())

    def get_batch_loss(self):
        return self.batch_loss
    def get_name_value(self):
        total_cer = float(self.total_l_dist) / float(self.total_n_label)
        return total_cer, self.total_n_label, self.total_l_dist, self.total_ctc_loss
    def reset(self):
        self.total_n_label = 0
        self.total_l_dist = 0
        self.num_inst = 0
        self.sum_metric = 0.0
        self.total_ctc_loss = 0.0

loss_metric =LossMetric(batch_size, 1)
mlp_model = mx.mod.Module(symbol=final_net, context=mx.cpu(), data_names=('data',), label_names=('softmax_label',))
mlp_model.fit(train_iter,  # train data
              eval_data=val_iter,  # validation data
              optimizer='adam',  # use SGD to train
              optimizer_params={'learning_rate': 0.001},  # use fixed learning rate
              eval_metric=loss_metric,  # report accuracy during training
              num_epoch=40)  # train for at most 10 dataset passes
