from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        Z_max = Z.max(axis=-1, keepdims=True)
        Z_norm = Z - Z_max # use Z - max to compute below to prevent overflow
        log_sum_exp = array_api.log(array_api.exp(Z_norm).sum(axis=-1, keepdims=True))
        log_softmax = Z_norm - log_sum_exp
        return log_softmax
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        xp = -exp(node) + 1
        grad = out_grad * (-exp(node) + 1)
        z_shape_reduction = list(z.shape)
        z_shape_reduction[-1] = 1
        out_grad_ndim = len(out_grad.shape)
        out_grad_sum = out_grad.sum((out_grad_ndim - 1, )).reshape(tuple(z_shape_reduction)).broadcast_to(z.shape)
        grad = out_grad - out_grad_sum * exp(node)
        return grad
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION

        # broadcast_to cannot process new_shape's ndim doesn't equal to old_shape's ndim
        if self.axes is None:
            reshape_broad = list(Z.shape)
            for i in range(len(reshape_broad)):
                reshape_broad[i] = 1
            max_Z_origin = Z.max().reshape(reshape_broad).broadcast_to(Z.shape)
        else:
            max_Z_origin = Z.max(self.axes, keepdims=True).broadcast_to(Z.shape)

        max_Z_reduce = Z.max(self.axes)
        exp_shifted = array_api.exp(Z - max_Z_origin)
        sum_exp = exp_shifted.sum(self.axes)
        log_sum_exp = array_api.log(sum_exp) + max_Z_reduce

        return log_sum_exp
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
    
        z = node.inputs[0]
        expand_axes = list(z.shape)

        if self.axes != None:
            if isinstance(self.axes, tuple):
                for axis in list(self.axes):
                    expand_axes[axis] = 1
            else:
                expand_axes[self.axes] = 1
        else:
            for i in range(len(expand_axes)):
                expand_axes[i] = 1

        node = node.reshape(expand_axes).broadcast_to(z.shape)
        out_grad = out_grad.reshape(expand_axes).broadcast_to(z.shape)
        grad = exp((z - node)) * out_grad

        return grad
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

# Author: Qingzheng Wang