from typing import Tuple, Callable, List

import tensorflow as tf


def _f32(x):
    return tf.cast(x, tf.float32)


def get_sparsity_tensor_of_variable(variables: List[tf.Variable]) -> Tuple[tf.Tensor, List[tf.Tensor]]:

    weight_size = []
    sparsity_list = []

    total_sparsity = 0.

    for i in range(len(variables)):

        num_parameters = tf.reduce_prod(tf.shape(variables[i]))
        sparsity_of_layer = tf.reduce_mean(_f32(tf.equal(variables[i], 0.)))

        weight_size.append(num_parameters)
        sparsity_list.append(sparsity_of_layer)

        total_sparsity = total_sparsity + _f32(num_parameters) * sparsity_of_layer

    total_sparsity = total_sparsity / _f32(tf.reduce_sum(weight_size))

    return total_sparsity, sparsity_list


def get_sparsity_of_variable(sess: tf.Session,
                             variables: List[tf.Variable] = None,
                             variable_filter: Callable = None) -> Tuple[float, List[float]]:

    assert not (variables is None and variable_filter is None)

    if variable_filter is not None:
        variables = [svar for svar in tf.trainable_variables() if variable_filter(svar.name)]

    total_sparsity_tensor, sparsity_tensor_list = get_sparsity_tensor_of_variable(variables)
    return sess.run(total_sparsity_tensor), sess.run(sparsity_tensor_list)


def cges(learning_rate: float,
         lamb: float,
         mu: float,
         chvar: float,
         group_layerwise: List[float],
         exclusive_layerwise: List[float],
         variable_filter: Callable = None) -> Tuple[tf.Operation, list]:
    """
    :param learning_rate: learning rate
    :param lamb: parameter that decides the entire regularization effect
    :param mu: parameter for balancing the sharing and competition term at each layer.
    :param chvar: mu change per layer
    :param group_layerwise: e.g. [1., 1.0, 1. / 15, 1. / 144]
    :param exclusive_layerwise: e.g. [1., 0.5, 15., 144.]
    :param variable_filter: function for filtering variables
    :return:
    """
    variable_filter = variable_filter or (lambda x: True)
    S_vars = [svar for svar in tf.trainable_variables() if variable_filter(svar.name)]
    assert len(S_vars) == len(group_layerwise), "len(S_vars): {} is not len(group_layerwise): {}".format(
        len(S_vars), len(group_layerwise))
    assert len(S_vars) == len(exclusive_layerwise), "len(S_vars): {} is not len(exclusive_layerwise): {}".format(
        len(S_vars), len(exclusive_layerwise))

    op_list = []
    for var_idx, var in enumerate(S_vars):

        # GS
        group_sum = tf.reduce_sum(tf.square(var), -1)  # (w, h, in)
        g_param = learning_rate * lamb * (mu - var_idx * chvar)
        gl_comp = 1. - g_param * group_layerwise[var_idx] * tf.rsqrt(group_sum)  # (w, h, in)
        gl_plus = tf.cast(gl_comp > 0, tf.float32) * gl_comp  # (w, h, in)
        gl_stack = tf.stack([gl_plus for _ in range(var.get_shape()[-1])], -1)  # (w, h, in, out)
        gl_op = gl_stack * var  # (w, h, in, out)

        # ES
        e_param = learning_rate * lamb * ((1. - mu) + var_idx * chvar)
        W_sum = e_param * exclusive_layerwise[var_idx] * tf.reduce_sum(tf.abs(gl_op), -1)
        W_sum_stack = tf.stack([W_sum for _ in range(gl_op.get_shape()[-1])], -1)
        el_comp = tf.abs(gl_op) - W_sum_stack
        el_plus = tf.cast(el_comp > 0, tf.float32) * el_comp
        cges_op = var.assign(el_plus * tf.sign(gl_op))
        op_list.append(cges_op)

    with tf.control_dependencies(op_list):
        cges_op_list = tf.no_op()

    return cges_op_list, op_list
