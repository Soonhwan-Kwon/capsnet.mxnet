import mxnet as mx


def squash(data, squash_axis, name=''):
    epsilon = 1e-08
    s_squared_norm = mx.sym.sum(data=mx.sym.square(data, name='square_'+name), axis=squash_axis, keepdims=True, name='s_squared_norm_'+name)
    scale = s_squared_norm / (1 + s_squared_norm) / mx.sym.sqrt(data=(s_squared_norm+epsilon), name='s_squared_norm_sqrt_'+name)
    squashed_net = mx.sym.broadcast_mul(scale, data, name='squashed_net_'+name)
    return squashed_net


def PrimaryCaps(data, dim_vector, n_channels, kernel_size, strides, name=''):
    """
    :param data: 4D tensor of shape [batch_size, width, height, channels]
    :param dim_vector:
    :param n_channels:
    :param kernel_size:
    :param strides:
    :param name:
    :return: 3D tensor of shape [batch_size, num_capsule, dim_vector]
    """
    out = mx.sym.Convolution(data=data,
                             num_filter=dim_vector * n_channels,
                             kernel=kernel_size,
                             stride=strides,
                             name=name
                             )
    # print('after conv2d 2',primarycaps.infer_shape(data=(batch_size, 1, 28, 28)))
    # net.shape = [batch_size, 1152,8]
    out = mx.sym.Reshape(data=out, shape=(0, -1, dim_vector))
    out = squash(out, squash_axis=2)
    # primarycaps.infer_shape(data=(batch_size, 1, 28, 28))
    # print('after squash', primarycaps.infer_shape(data=(batch_size, 1, 28, 28)))
    return out


class CapsuleLayer():
    """
    The capsule layer with dynamic routing.
    [batch_size, input_num_capsule, input_dim_vector] => [batch_size, num_capsule, dim_vector]
    """

    def __init__(self, num_capsule, dim_vector, batch_size, kernel_initializer, bias_initializer, num_routing=3):
        self.num_capsule = num_capsule
        self.dim_vector = dim_vector
        self.batch_size = batch_size
        self.num_routing = num_routing
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def __call__(self, data):
        _, out_shapes, __ = data.infer_shape(data=(self.batch_size, 1, 28, 28))
        _, input_num_capsule, input_dim_vector = out_shapes[0]

        # build w and bias
        # ('W', TensorShape([Dimension(1152), Dimension(10), Dimension(8), Dimension(16)]))
        # ('B', TensorShape([Dimension(1), Dimension(1152), Dimension(10), Dimension(1), Dimension(1)]))
        w = mx.sym.Variable('Weight',
                            shape=(1, input_num_capsule, self.num_capsule, input_dim_vector, self.dim_vector),
                            init=self.kernel_initializer)
        bias = mx.sym.Variable('Bias',
                               shape=(self.batch_size, input_num_capsule, self.num_capsule, 1, 1),
                               init=self.bias_initializer)

        # ('input', TensorShape([Dimension(None), Dimension(1152), Dimension(8)]))
        # ('inputs_expand', TensorShape([Dimension(None), Dimension(1152), Dimension(1), Dimension(8), Dimension(1)]))
        inputs_expand = mx.sym.Reshape(data=data, shape=(0, 0, -4, -1, 1))
        inputs_expand = mx.sym.Reshape(data=inputs_expand, shape=(0, 0, -4, 1, -1, 0))
        print('after inputs_expand', inputs_expand.infer_shape(data=(self.batch_size, 1, 28, 28)))
        # input_tiled
        # ('inputs_tiled.shape', TensorShape([Dimension(None), Dimension(1152), Dimension(10), Dimension(8), Dimension(1)]))
        inputs_tiled = mx.sym.tile(data=inputs_expand, reps=(1, 1, self.num_capsule, 1, 1))
        inputs_tiled.infer_shape(data=(self.batch_size, 1, 28, 28))
        # w_tiled
        # w_tiled.shape = [(1L, 1152L, 10L, 8L, 16L)]
        w_tiled = mx.sym.tile(w, reps=(self.batch_size, 1, 1, 1, 1))
        w_tiled.infer_shape()
        inputs_hat = mx.sym.linalg_gemm2(w_tiled, inputs_tiled, transpose_a=True)
        inputs_hat = mx.sym.swapaxes(data=inputs_hat, dim1=3, dim2=4)
        print('inputs_hat', inputs_hat.infer_shape(data=(self.batch_size, 1, 28, 28)))
        # ('i', 0, 'c', TensorShape([Dimension(1), Dimension(1152), Dimension(10), Dimension(1), Dimension(1)]))

        for i in range(0, self.num_routing):
            c = mx.sym.softmax(bias, axis=2, name='c' + str(i))
            # print('i',i,' c',c.infer_shape())
            outputs = squash(
                mx.sym.sum(mx.sym.broadcast_mul(c, inputs_hat, name='broadcast_mul_' + str(i)), axis=1, keepdims=True,
                           name='sum_' + str(i)), name='output_' + str(i), squash_axis=4)
            # print('i', i, ' output', outputs.infer_shape(data=(1, 28, 28)))
            if i != self.num_routing - 1:
                bias = bias + mx.sym.sum(mx.sym.broadcast_mul(c, inputs_hat, name='bias_broadcast_mul' + str(i)), axis=4,
                                         keepdims=True, name='bias_' + str(i))

        # ('digitcaps', TensorShape([Dimension(None), Dimension(10), Dimension(16)]))
        outputs = mx.sym.Reshape(data=outputs, shape=(-1, self.num_capsule, self.dim_vector))
        outputs.infer_shape(data=(self.batch_size, 1, 28, 28))
        # ('i', 0, 'outputs', TensorShape([Dimension(None), Dimension(1), Dimension(10), Dimension(1), Dimension(16)]))
        # ('bias', TensorShape([Dimension(None), Dimension(1152), Dimension(10), Dimension(1), Dimension(1)]))
        # ('i', 1, 'c', TensorShape([Dimension(None), Dimension(1152), Dimension(10), Dimension(1), Dimension(1)]))
        # ('i', 1, 'outputs', TensorShape([Dimension(None), Dimension(1), Dimension(10), Dimension(1), Dimension(16)]))
        # ('bias', TensorShape([Dimension(None), Dimension(1152), Dimension(10), Dimension(1), Dimension(1)]))
        # ('i', 2, 'c', TensorShape([Dimension(None), Dimension(1152), Dimension(10), Dimension(1), Dimension(1)]))
        # ('i', 2, 'outputs', TensorShape([Dimension(None), Dimension(1), Dimension(10), Dimension(1), Dimension(16)]))
        return outputs
