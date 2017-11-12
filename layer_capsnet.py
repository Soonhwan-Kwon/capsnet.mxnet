import mxnet as mx
import numpy as np
data = mx.sym.Variable('data')
conv1_num_filter = 256
kernel = (9, 9)
batch_size = 1
epsilon = 1e-08
input_shape =(1, 28, 28)
#Conv2D layer
# net.shape = [batch_size, 256, 20, 20]
net = mx.sym.Reshape(data=data, shape=(-4, -1, 1, 0, 0))
net.infer_shape(data=(1, 28, 28))
net = mx.sym.Convolution(data=net,
                         num_filter=conv1_num_filter,
                         kernel=kernel,
                         layout='NCHW',
                         name='conv1')
net = mx.sym.Activation(data=net, act_type='relu', name='conv1_act')
net.infer_shape(data=(1, 28, 28))
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
net.infer_shape(data=(1, 28, 28))
# net.shape = [batch_size, 1152,8]
net = mx.sym.Reshape(data=net, shape=(0, -1, dim_vector))


def squash(net, name=''):
    s_squared_norm = mx.sym.sum(data=mx.sym.square(net, name='square_'+name), axis=squash_axis, keepdims=True, name='s_squared_norm_'+name)
    scale = s_squared_norm / (1 + s_squared_norm) / mx.sym.sqrt(data=(s_squared_norm+epsilon), name='s_squared_norm_sqrt_'+name)
    squashed_net = mx.sym.broadcast_mul(scale, net, name='squashed_net_'+name)
    return squashed_net

net = squash(net)
net.infer_shape(data=(1, 28, 28))
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
                       shape=(1, input_num_capsule, num_capsule, 1, 1),
                       init=bias_initializer)

# ('input', TensorShape([Dimension(None), Dimension(1152), Dimension(8)]))
# ('inputs_expand', TensorShape([Dimension(None), Dimension(1152), Dimension(1), Dimension(8), Dimension(1)]))
net = mx.sym.Reshape(data=net, shape=(0, 0, -4, -1, 1))
net = mx.sym.Reshape(data=net, shape=(0, 0, -4, 1, -1, 0))
net.infer_shape(data=(1, 28, 28))
# input_tiled
# ('inputs_tiled.shape', TensorShape([Dimension(None), Dimension(1152), Dimension(10), Dimension(8), Dimension(1)]))
net = mx.sym.tile(data=net, reps=(1, 1, n_class, 1, 1))
net.infer_shape(data=(1, 28, 28))
# w_tiled
# w_tiled.shape = [(1L, 1152L, 10L, 8L, 16L)]
w_tiled = mx.sym.tile(w, reps=(batch_size, 1, 1, 1, 1))
w_tiled.infer_shape()
#w_tiled_transposed = mx.sym.transpose(w_tiled)
# Compute 'inputs * W' by scanning inputs_tiled on dimension 0
# input_hat.shape = [None, intput_num_capsule, num_capsule, 1, dim_vector]
# ('inputs_hat', TensorShape([Dimension(None), Dimension(1152), Dimension(10), Dimension(1), Dimension(16)]))
#net = mx.sym.transpose(net)
inputs_hat = mx.sym.linalg_gemm(w_tiled, net, transpose_a=True)
inputs_hat = mx.sym.swapaxes(data=inputs_hat, dim1=3, dim2=4)
inputs_hat.infer_shape(data=(1, 28, 28))
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
digitcaps.infer_shape(data=(1, 28, 28))
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
out_caps.infer_shape(data=(1, 28, 28))
y = mx.sym.Variable('y', shape=(batch_size, n_class))
y = mx.sym.Reshape(data=y, shape=(batch_size, -4, n_class, -1))
y.infer_shape(y=(1, 10))
inputs_masked = mx.sym.linalg_gemm(mx.sym.transpose(y), digitcaps, transpose_a=True)
# y.infer_shape(y=(1,10))
# inputs_masked = mx.sym.linalg_gemm(out_caps, y, transpose_a=True)
#inputs_masked = mx.sym.batch_dot(out_caps, y)
print(inputs_masked.infer_shape(data=(1, 28, 28), y=(1, 10)))
inputs_masked = mx.sym.Reshape(data=inputs_masked, shape=(-3, 0))
print(inputs_masked.infer_shape(data=(1, 28, 28), y=(1, 10)))
x_recon = mx.sym.FullyConnected(data=inputs_masked, num_hidden=512, name='x_recon')
x_recon = mx.sym.Activation(data=x_recon, act_type='relu', name='x_recon_act')
x_recon = mx.sym.FullyConnected(data=x_recon, num_hidden=1024, name='x_recon2')
x_recon = mx.sym.Activation(data=x_recon, act_type='relu', name='x_recon_act2')

x_recon = mx.sym.FullyConnected(data=x_recon, num_hidden=np.prod(input_shape), name='x_recon3')
x_recon = mx.sym.Activation(data=x_recon, act_type='relu', name='x_recon_act3')

