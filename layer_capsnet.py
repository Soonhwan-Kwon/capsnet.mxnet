import mxnet as mx
data = mx.sym.Variable('data')
label = mx.sym.Variable('label')
conv1_num_filter = 256
kernel = (9, 9)
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

# net.shape = [batch_size, 256, 6, 6]
dim_vector = 8
n_channels = 32
squash_axis = 2
net = mx.sym.Convolution(data=net,
                         num_filter=dim_vector*n_channels,
                         kernel=kernel,
                         )

# net.shape = [batch_size, 1152,8]
net = mx.sym.Reshape(data=net, shape=(0, -1, dim_vector))

def squash(net):
    s_squared_norm = mx.sym.sum(data=mx.sym.square(net), axis=squash_axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / mx.sym.sqrt(data=(s_squared_norm+1e-08))
    squashed_net = mx.sym.broadcast_mul(scale, net)
    return squashed_net

net = squash(net)

n_class = 10
input_num_capsule = n_class
input_dim_vector = dim_vector
dim_vector = 16
num_routing = 3
# CapsuleLayer

# int_num_capsule = n_class
# input_shape = [None, input_num_capsule
# input_expand
net = mx.sym.Reshape(data=net, shape=(0, 0, -4, 1, -1))
net = mx.sym.Reshape(data=net, shape=(0, 0, -4, 1, -1, 0))
# input_tiled
net = mx.sym.tile(data=net, reps=(1, 1, n_class, 1, 1))
# w_tiled
W = mx.sym.Variable('Weight',
                    shape=(n_class, input_num_capsule, input_dim_vector, 16),
                    init=mx.init.Xavier(rnd_type='uniform', factor_type='avg', magnitude=3))
W = mx.sym.Variable('Bias',
                    shape=(n_class, input_num_capsule, input_dim_vector, 16),
                    init=mx.init.Xavier(rnd_type='uniform', factor_type='avg', magnitude=3))
net = mx.sym.Rehape()