'''
Reproducing paper:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"
'''
import mxnet as mx

batch_size = 128
I1, I2, I3, I4, J1, J2, J3, J4 = 8, 8, 4, 4, 8, 8, 4, 4
stddev = 0.5
Rank = 2


def tucker_fc(input_layer):
    W1 = mx.symbol.Variable(name='W1', shape=(I1 * J1 * Rank, 1), init=mx.init.Normal(stddev))
    W2 = mx.symbol.Variable(name='W2', shape=(1, I2 * J2 * Rank), init=mx.init.Normal(stddev))
    W3 = mx.symbol.Variable(name='W3', shape=(Rank * I3 * J3, 1), init=mx.init.Normal(stddev))
    W4 = mx.symbol.Variable(name='W4', shape=(1, Rank * I4 * J4), init=mx.init.Normal(stddev))
    W_core = mx.symbol.Variable(name='W_core', shape=(Rank ** 2, Rank ** 2), init=mx.init.Normal(stddev))
    # multiply W1 and W2 and W_core
    s1_tmp = mx.symbol.dot(W1, W2)
    s1_tmp = mx.symbol.reshape(s1_tmp, shape=(I1, J1 * Rank, I2, J2 * Rank))
    s1_tmp = mx.symbol.transpose(s1_tmp, axes=(0, 2, 1, 3))
    s1_tmp = mx.symbol.reshape(s1_tmp, shape=(I1 * I2 * J1, Rank, J2, Rank))
    s1_tmp = mx.symbol.transpose(s1_tmp, axes=(0, 2, 1, 3))
    s1_tmp = mx.symbol.reshape(s1_tmp, shape=(I1 * I2 * J1 * J2, Rank * Rank))
    s1_tmp = mx.symbol.dot(s1_tmp, W_core)
    s1_tmp = mx.symbol.reshape(s1_tmp, shape=(I1 * I2, J1 * J2 * Rank * Rank))
    # multiply InData
    s2_tmp = mx.symbol.reshape(input_layer, shape=(batch_size, I1 * I2, I3 * I4))
    s2_tmp = mx.symbol.transpose(s2_tmp, axes=(0, 2, 1))
    s2_tmp = mx.symbol.reshape(s2_tmp, shape=(batch_size * I3 * I4, I1 * I2))
    s2_tmp = mx.symbol.dot(s2_tmp, s1_tmp)
    s2_tmp = mx.symbol.reshape(s2_tmp, shape=(batch_size, I3 * I4, J1 * J2 * Rank * Rank))
    s2_tmp = mx.symbol.transpose(s2_tmp, axes=(0, 2, 1))
    s2_tmp = mx.symbol.reshape(s2_tmp, shape=(batch_size * J1 * J2, Rank * Rank * I3 * I4))
    # multiply W3 and W4
    s3_tmp = mx.symbol.dot(W3, W4)
    s3_tmp = mx.symbol.reshape(s3_tmp, shape=(Rank, I3 * J3, Rank, I4 * J4))
    s3_tmp = mx.symbol.transpose(s3_tmp, axes=(0, 2, 1, 3))
    s3_tmp = mx.symbol.reshape(s3_tmp, shape=(Rank * Rank * I3, J3, I4, J4))
    s3_tmp = mx.symbol.transpose(s3_tmp, axes=(0, 2, 1, 3))
    s3_tmp = mx.symbol.reshape(s3_tmp, shape=(Rank * Rank * I3 * I4, J3 * J4))
    # multiply InData and W34
    OutData = mx.symbol.reshape(mx.symbol.dot(s2_tmp, s3_tmp), shape=(batch_size, J1 * J2 * J3 * J4))
    return OutData


def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=512,
                  memonger=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tupe
        Stride used in convolution
    dim_match : Boolen
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1),
                                   pad=(0, 0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride,
                                   pad=(1, 1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                          workspace=workspace, name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                          workspace=workspace, name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut


def resnet(units, num_stage, filter_list, num_class, data_type, bottle_neck=True, bn_mom=0.9, workspace=512,
           memonger=False):
    """Return ResNet symbol of cifar10 and imagenet
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stage : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_class : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert (num_unit == num_stage)
    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    if data_type == 'cifar10':
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                  no_bias=True, name="conv0", workspace=workspace)
    elif data_type == 'imagenet':
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')
    else:
        raise ValueError("do not support {} yet".format(data_type))
    for i in range(num_stage):
        body = residual_unit(body, filter_list[i + 1], (1 if i == 0 else 2, 1 if i == 0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i] - 1):
            body = residual_unit(body, filter_list[i + 1], (1, 1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')

    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.symbol.Pooling(data=relu1, kernel=(2, 2), stride=(2,2), pool_type='max', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    #
    # arg_shape, out_shape, aux_shape = flat.infer_shape(data=(128, 3,32,32))
    # print(out_shape)

    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=64 * 4 * 4, weight = mx.symbol.Variable(name='fc_weight', init=mx.init.Normal(0.005)),name='fc1')
    # fc1 = tucker_fc(flat)
    bn2 = mx.sym.BatchNorm(data=fc1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn2')

    fc2 = mx.symbol.FullyConnected(data=bn2, num_hidden=num_class, name='fc2')

    return mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
