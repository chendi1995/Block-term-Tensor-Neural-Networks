# Created by ay27 at 05/11/2017
import numpy as np
from keras import backend as K
import tensorflow as tf


def split_kernel_into_core_and_factors(kernel, input_shape, output_shape, core_ranks, block_ranks):
    # the kernel layout is : [[core, factor0, factor1, factor2, ...],
    #                         [core, factor0, factor1, factor2, ...],
    #                         ...]
    one_tucker_size = np.sum(input_shape * output_shape * core_ranks) + np.prod(core_ranks)
    cores = [kernel[(_ * one_tucker_size): (_ * one_tucker_size + np.prod(core_ranks))]
             for _ in range(block_ranks)]
    for _ in range(block_ranks):
        cores[_] = K.reshape(cores[_], core_ranks)

    one_tucker_shape = input_shape * output_shape * core_ranks
    factors = []
    for ii in range(block_ranks):
        start_idx = ii * one_tucker_size + np.prod(core_ranks)
        factors.append([])
        for jj in range(len(core_ranks)):
            s = start_idx + np.sum(one_tucker_shape[:jj])
            e = start_idx + np.sum(one_tucker_shape[:(jj + 1)])
            tmp = kernel[s:e]
            tmp = K.reshape(tmp, [input_shape[jj], output_shape[jj], core_ranks[jj]])
            factors[ii].append(tmp)
    return cores, factors


def BT_mul2(x, cores, factors, in_shape, out_shape, core_shape):
    """
       Perform y = xW, while the W is represented by a block-term tucker decomposition.
       For example, W = [core, factor] + [core, factor] + ...

       Parameters
       ----------
       x : Tensor
           input data, shape is (batch_size, dim)
       cores : list
           list of core tensors, shape is (block_rank, core_ranks)
       factors : list
           list of factor tensors, shape is (block_rank, input_dim, output_dim, core_rank)
       in_shape : list
       out_shape : list
       core_shape : list
       verbose : bool

       Returns
       -------
       Tensor
           y = xW = x[cores, factors]

       """
    assert isinstance(cores, list)
    assert isinstance(factors, list)
    assert isinstance(in_shape, np.ndarray)
    assert isinstance(out_shape, np.ndarray)
    assert isinstance(core_shape, np.ndarray)
    assert len(cores) == len(factors)  # check block_ranks

    assert len(cores[0].shape) == len(factors[0]) == 2
    I1, I2 = in_shape
    J1, J2 = out_shape
    R1, R2 = core_shape

    res = None

    for ii in range(len(cores)):
        W1 = factors[ii][0]
        W2 = factors[ii][1]
        g = cores[ii]

        W1 = K.reshape(W1, [I1 * J1 * R1, 1])
        W2 = K.reshape(W2, [1, I2 * J2 * R2])
        g = K.reshape(g, [R1 * R2, 1])

        s1_tmp = K.dot(W1, W2)
        s1_tmp = K.reshape(s1_tmp, [I1, J1, R1, I2, J2, R2])
        s1_tmp = K.permute_dimensions(s1_tmp, [0, 3, 1, 4, 2, 5])  # I1 I2 J1 J2 R1 R2
        s1_tmp = K.reshape(s1_tmp, [I1 * I2, J1 * J2 * R1 * R2])
        s1_tmp = K.dot(x, s1_tmp)  # batch_size, J1*J2*R1*R2
        s1_tmp = K.reshape(s1_tmp, [-1, R1 * R2])  # batch_size * J1 * J2

        s2_tmp = K.dot(s1_tmp, g)
        s2_tmp = K.reshape(s2_tmp, [-1, J1 * J2])
        if res is None:
            res = s2_tmp
        else:
            res += s2_tmp
    return res


def BT_mul3(x, cores, factors, in_shape, out_shape, core_shape):
    """
       Perform y = xW, while the W is represented by a block-term tucker decomposition.
       For example, W = [core, factor] + [core, factor] + ...

       Parameters
       ----------
       x : Tensor
           input data, shape is (batch_size, dim)
       cores : list
           list of core tensors, shape is (block_rank, core_ranks)
       factors : list
           list of factor tensors, shape is (block_rank, input_dim, output_dim, core_rank)
       in_shape : list
       out_shape : list
       core_shape : list

       Returns
       -------
       Tensor
           y = xW = x[cores, factors]

       """
    assert isinstance(cores, list)
    assert isinstance(factors, list)
    assert isinstance(in_shape, np.ndarray)
    assert isinstance(out_shape, np.ndarray)
    assert isinstance(core_shape, np.ndarray)
    assert len(cores) == len(factors)  # check block_ranks

    assert len(cores[0].shape) == len(factors[0]) == 3

    I1, I2, I3 = in_shape
    J1, J2, J3 = out_shape
    R1, R2, R3 = core_shape

    res = None
    for ii in range(len(cores)):
        W1 = factors[ii][0]
        W2 = factors[ii][1]
        W3 = factors[ii][2]
        g = cores[ii]

        W1 = K.reshape(W1, [int(np.prod(W1.shape)), 1])  # [I*N*R1, 1]
        W2 = K.reshape(W2, [1, int(np.prod(W2.shape))])  # [1, J*M*R2]
        W3 = K.reshape(W3, [I3 * R3, J3])
        g = K.reshape(g, [R1 * R2, R3])

        s1_tmp = K.dot(W1, W2)
        s1_tmp = K.reshape(s1_tmp, [I1, J1, R1, I2, J2, R2])
        s1_tmp = K.permute_dimensions(s1_tmp, [0, 3, 1, 4, 2, 5])
        s1_tmp = K.reshape(s1_tmp, [I1 * I2 * J1 * J2, R1 * R2])
        s1_tmp = K.dot(s1_tmp, g)
        s1_tmp = K.reshape(s1_tmp, [I1 * I2, J1 * J2 * R3])

        # multiply InData
        s2_tmp = K.reshape(x, [-1, I1 * I2, I3])
        s2_tmp = K.permute_dimensions(s2_tmp, [0, 2, 1])
        s2_tmp = K.reshape(s2_tmp, [-1, I1 * I2])  # batch_size * I3
        s2_tmp = K.dot(s2_tmp, s1_tmp)
        s2_tmp = K.reshape(s2_tmp, [-1, I3, J1 * J2 * R3])
        s2_tmp = K.permute_dimensions(s2_tmp, [0, 2, 1])
        s2_tmp = K.reshape(s2_tmp, [-1, R3 * I3])  # batch_size * J1 * J2

        # multiply InData and W3
        s3_tmp = K.reshape(W3, [I3 * R3, J3])

        out_tmp = K.dot(s2_tmp, s3_tmp)
        if res is None:
            res = K.reshape(out_tmp, [-1, J1 * J2 * J3])
        else:
            res += K.reshape(out_tmp, [-1, J1 * J2 * J3])
        return res


def BT_mul4(x, cores, factors, in_shape, out_shape, core_shape, verbose=False):
    """
    Perform y = xW, while the W is represented by a block-term tucker decomposition.
    For example, W = [core, factor] + [core, factor] + ...

    Parameters
    ----------
    x : Tensor
        input data, shape is (batch_size, dim)
    cores : list
        list of core tensors, shape is (block_rank, core_ranks)
    factors : list
        list of factor tensors, shape is (block_rank, input_dim, output_dim, core_rank)
    in_shape : list
    out_shape : list
    core_shape : list
    verbose : bool

    Returns
    -------
    Tensor
        y = xW = x[cores, factors]

    """
    assert isinstance(cores, list)
    assert isinstance(factors, list)
    assert isinstance(in_shape, np.ndarray)
    assert isinstance(out_shape, np.ndarray)
    assert isinstance(core_shape, np.ndarray)
    assert len(cores) == len(factors)  # check block_ranks

    assert len(cores[0].shape) == len(factors[0]) == 4  # check core_ranks

    I1, I2, I3, I4 = in_shape
    J1, J2, J3, J4 = out_shape
    R1, R2, R3, R4 = core_shape

    max_immd_size = 0

    res = None
    for ii in range(len(cores)):
        W1 = factors[ii][0]  # [I1, J1, R1]
        W2 = factors[ii][1]  # [I2, J2, R2]
        W3 = factors[ii][2]  # [I3, J3, R3]
        W4 = factors[ii][3]  # [L, Q, R4]
        g = cores[ii]  # [R1, R2, R3, R4]

        W1 = K.reshape(W1, [int(np.prod(W1.shape)), 1])  # [I1*J1*R1, 1]
        W2 = K.reshape(W2, [1, int(np.prod(W2.shape))])  # [1, I2*J2*R2]
        W3 = K.reshape(W3, [int(np.prod(W3.shape)), 1])  # [I3*J3*R3, 1]
        W4 = K.reshape(W4, [1, int(np.prod(W4.shape))])  # [1, I4*J4*R4]
        g = K.reshape(g, [R1 * R2, R3 * R4])  # [R1*R2, R3*R4]

        # First, mul W1 & W2 & core
        s1_tmp = K.dot(W1, W2)

        if verbose:
            max_immd_size = max(max_immd_size, int(np.prod(s1_tmp.shape)))

        s1_tmp = K.reshape(s1_tmp, [I1, J1, R1, I2, J2, R2])
        s1_tmp = K.permute_dimensions(s1_tmp, [0, 3, 1, 4, 2, 5])
        s1_tmp = K.reshape(s1_tmp, [I1 * I2 * J1 * J2, R1 * R2])
        s1_tmp = K.dot(s1_tmp, g)

        if verbose:
            max_immd_size = max(max_immd_size, int(np.prod(s1_tmp.shape)))

        s1_tmp = K.reshape(s1_tmp, [I1 * I2, J1 * J2 * R3 * R4])

        # Second, mul input data x
        s2_tmp = K.reshape(x, [-1, I1 * I2, I3 * I4])
        s2_tmp = K.permute_dimensions(s2_tmp, [0, 2, 1])
        s2_tmp = K.reshape(s2_tmp, [-1, I1 * I2])  # batch_size * I3 * I4
        s2_tmp = K.dot(s2_tmp, s1_tmp)

        if verbose:
            max_immd_size = max(max_immd_size, int(np.prod(s2_tmp.shape)))

        s2_tmp = K.reshape(s2_tmp, [-1, I3 * I4, J1 * J2 * R3 * R4])
        s2_tmp = K.permute_dimensions(s2_tmp, [0, 2, 1])
        s2_tmp = K.reshape(s2_tmp, [-1, R3 * R4 * I3 * I4])  # batch_size * J1 * J2

        # Third, mul W3 & W4
        s3_tmp = K.dot(W3, W4)

        if verbose:
            max_immd_size = max(max_immd_size, int(np.prod(s3_tmp.shape)))

        s3_tmp = K.reshape(s3_tmp, [I3, J3, R3, I4, J4, R4])
        s3_tmp = K.permute_dimensions(s3_tmp, [2, 5, 0, 3, 1, 4])
        s3_tmp = K.reshape(s3_tmp, [R3 * R4 * I3 * I4, J3 * J4])

        out_tmp = K.dot(s2_tmp, s3_tmp)

        if verbose:
            max_immd_size = max(max_immd_size, int(np.prod(out_tmp.shape)))

        out_tmp = K.reshape(out_tmp, [-1, int(np.prod(out_shape))])  # [batch, N*M*P*Q]
        if res is None:
            res = out_tmp
        else:
            res += out_tmp

    if verbose:
        print('max immediate size is : %d, compress ratio : %f' %
              (max_immd_size * len(cores), np.prod(in_shape) * np.prod(out_shape) / (max_immd_size * len(cores))))
    return res


def BT_mul5(x, cores, factors, in_shape, out_shape, core_shape, verbose=False):
    """
    Perform y = xW, while the W is represented by a block-term tucker decomposition.
    For example, W = [core, factor] + [core, factor] + ...

    Parameters
    ----------
    x : Tensor
        input data, shape is (batch_size, dim)
    cores : list
        list of core tensors, shape is (block_rank, core_ranks)
    factors : list
        list of factor tensors, shape is (block_rank, input_dim, output_dim, core_rank)
    in_shape : list
    out_shape : list
    core_shape : list
    verbose : bool

    Returns
    -------
    Tensor
        y = xW = x[cores, factors]

    """
    assert isinstance(cores, list)
    assert isinstance(factors, list)
    assert isinstance(in_shape, np.ndarray)
    assert isinstance(out_shape, np.ndarray)
    assert isinstance(core_shape, np.ndarray)
    assert len(cores) == len(factors)  # check block_ranks

    assert len(cores[0].shape) == len(factors[0]) == 5  # check core_ranks

    I1, I2, I3, I4, I5 = in_shape
    J1, J2, J3, J4, J5 = out_shape
    R1, R2, R3, R4, R5 = core_shape

    res = None
    for ii in range(len(cores)):
        W1 = factors[ii][0]
        W2 = factors[ii][1]
        W3 = factors[ii][2]
        W4 = factors[ii][3]
        W5 = factors[ii][4]
        g = cores[ii]

        W1 = K.reshape(W1, [int(np.prod(W1.shape)), 1])
        W2 = K.reshape(W2, [1, int(np.prod(W2.shape))])
        W3 = K.reshape(W3, [int(np.prod(W3.shape)), 1])
        W4 = K.reshape(W4, [1, int(np.prod(W4.shape))])
        W5 = K.reshape(W5, [I5, R5, J5])
        g = K.reshape(g, [R1 * R2, R3 * R4 * R5])

        s1_tmp = K.dot(W1, W2)
        s1_tmp = K.reshape(s1_tmp, [I1, J1, R1, I2, J2, R2])
        s1_tmp = K.permute_dimensions(s1_tmp, [0, 3, 1, 4, 2, 5])
        s1_tmp = K.reshape(s1_tmp, [I1 * I2 * J1 * J2, R1 * R2])
        s1_tmp = K.dot(s1_tmp, g)
        s1_tmp = K.reshape(s1_tmp, [I1 * I2, J1 * J2 * R3 * R4 * R5])
        # multiply InData
        s2_tmp = K.reshape(x, [-1, I1 * I2, I3 * I4 * I5])
        s2_tmp = K.permute_dimensions(s2_tmp, [0, 2, 1])
        s2_tmp = K.reshape(s2_tmp, [-1, I1 * I2])  # batch_size * I3 * I4 * I5
        s2_tmp = K.dot(s2_tmp, s1_tmp)
        s2_tmp = K.reshape(s2_tmp, [-1, I3, I4, I5, J1, J2, R3, R4, R5])
        s2_tmp = K.permute_dimensions(s2_tmp, [0, 3, 4, 5, 8, 1, 2, 6, 7])
        s2_tmp = K.reshape(s2_tmp, [-1, I3 * I4 * R3 * R4])  # batch_size * I5 * J1 * J2 * R5
        # multiply W3 and W4
        s3_tmp = K.dot(W3, W4)
        s3_tmp = K.reshape(s3_tmp, [I3, J3, R3, I4, J4, R4])
        s3_tmp = K.permute_dimensions(s3_tmp, [0, 3, 2, 5, 1, 4])
        s3_tmp = K.reshape(s3_tmp, [I3 * I4 * R3 * R4, J3 * J4])
        s3_tmp = K.dot(s2_tmp, s3_tmp)  # [batch * I5 * J1 * J2 * R5, J3 * J4]
        # multiply W5
        s3_tmp = K.reshape(s3_tmp, [-1, I5, J1, J2, R5, J3, J4])
        s3_tmp = K.permute_dimensions(s3_tmp, [0, 2, 3, 5, 6, 1, 4])
        s3_tmp = K.reshape(s3_tmp, [-1, I5 * R5])
        W5 = K.reshape(W5, [I5 * R5, J5])
        out_tmp = K.dot(s3_tmp, W5)

        if res is None:
            res = tf.reshape(out_tmp, [-1, J1 * J2 * J3 * J4 * J5])
        else:
            res += tf.reshape(out_tmp, [-1, J1 * J2 * J3 * J4 * J5])
    return res
