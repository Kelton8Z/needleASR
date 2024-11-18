"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power.
    This operation will not be implemented in Needle NDArray. 
    """

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * self.scalar * a ** (self.scalar - 1) 
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        gradient_a = out_grad / b
        gradient_b = -out_grad * a / b ** 2
        return gradient_a, gradient_b
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        permutation = list(range(len(a.shape)))

        if self.axes is None: # Default to the last two axes
            temp = permutation[-1]
            permutation[-1] = permutation[-2]
            permutation[-2] = temp   
        else:
            permutation[self.axes[0]] = self.axes[1]
            permutation[self.axes[1]] = self.axes[0]

        return a.permute(tuple(permutation))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        permutation = list(range(len(out_grad.shape)))

        if self.axes == None: # Default to the last two axes
            axes = tuple(permutation[-2], permutation[-1])
            return transpose(out_grad, axes=axes)
        else:
            return transpose(out_grad, axes=self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


# Author: Qingzheng Wang
class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        a = a.compact()
        return a.reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        input_shape = a.shape
        return out_grad.reshape(input_shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)

# Author: Qingzheng Wang
class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.broadcast_to(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        input_shape = a.shape
        output_shape = self.shape
        broad_axes = []
        distance = len(output_shape) - len(input_shape)
        # input:     [3, 4] 
        # output: [2, 3, 5]
        # Add axis for 2:
        if distance > 0:
            broad_axes.extend(range(distance))
        # input:     [3, 4] 
        # output: [2, 3, 5]
        # Add axis for 5:
        for i, (m, n) in enumerate(zip(input_shape, output_shape[distance:])):
            if m != n:
                broad_axes.append(i)

        return out_grad.sum(tuple(broad_axes)).reshape(a.shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)

# Author: Qingzheng Wang
class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # Author: Qingzheng Wang
        if isinstance(self.axes, tuple): # e.g. (1, 2, 3)
            for axis in reversed(sorted(self.axes)):
                a = a.sum(axis)
        else: # axes could be a single int or None. 
            a = a.sum(self.axes)
        return a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Author: Qingzheng Wang
        a = node.inputs[0]
        squeezed_shape = list(a.shape)

        # The squeezed shape is the shape of the summed but keep_shape shape
        # if a (5, 4), axes=(1,), squeezed shape is (5, 1).
        #
        # If the self.axes == None, then out_grad is (1, ), this must reshape 
        # to with the same shape as a, but (1, 1)
        #
        # This is used for reshape out_grad and then broadcast it to 
        # the same shape as a.
        if self.axes != None:
            if isinstance(self.axes, tuple):
                for axis in self.axes:
                    squeezed_shape[axis] = 1
            else:
                squeezed_shape[self.axes] = 1
        else:
            for i in range(len(squeezed_shape)):
                squeezed_shape[i] = 1
        
        return out_grad.reshape(tuple(squeezed_shape)).broadcast_to(a.shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        gradient_a = matmul(out_grad, transpose(b))
        gradient_b = matmul(transpose(a), out_grad)

        # matmul will do broadcast on N-D arrays
        # so, here should "inverse broadcast" (reduce 
        # the broadcasted dimensions) to the original shape
        # e.g.
        #   gradient_a.shape    [3, 3,  2,  1]
        #   a.shape                [3,  2,  1]
        #   reduce_axes         [0]
        #   inverse broadcast to   [3,  2,  1]
        if gradient_a.shape != a.shape: 
            reduce_axes = []
            for i in range(-1, -len(a.shape) - 1, -1):
                if gradient_a.shape[i] != a.shape[i]:
                    reduce_axes.append(i + len(gradient_a.shape))
                if i == -len(a.shape) and i > -len(gradient_a.shape):
                    for j in range(-len(a.shape) - 1, -len(gradient_a.shape) - 1, -1):
                        reduce_axes.append(j + len(gradient_a.shape))
            gradient_a = gradient_a.sum(tuple(reduce_axes))

        if gradient_b.shape != b.shape:
            reduce_axes = []
            for i in range(-1, -len(b.shape) - 1, -1):
                if gradient_b.shape[i] != b.shape[i]:
                    reduce_axes.append(i + len(gradient_b.shape))
                if i == -len(b.shape) and i > -len(gradient_b.shape):
                    for j in range(-len(b.shape) - 1, -len(gradient_b.shape) - 1, -1):
                        reduce_axes.append(j + len(gradient_b.shape))
            gradient_b = gradient_b.sum(tuple(reduce_axes))

        return gradient_a, gradient_b
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.log()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad / a
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.exp()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * node
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.maximum(0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Since we would not compute second derivative of ReLU, so we can use NDArray here.
        a = node.inputs[0].realize_cached_data() # Transform Tensor to NDArray, the cached_data in Tensor is NDArray
        grad_mask = a > 0 # Then we can compute mask in NDArray
        return out_grad * Tensor(grad_mask, device=out_grad.device, dtype=out_grad.dtype) # But in coputaional graph we must use Tensor
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.tanh()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # tanh' = 1 - tanh^2
        return out_grad * (-node ** 2 + 1)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: Tuple[NDArray, ...]) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # Author: Qingzheng Wang
        assert len(args) > 0
        n = len(args)
        shape = args[0].shape
        shape_stack = list(shape)
        shape_stack.insert(self.axis, n) # insert n in a new dimension

        array_stack = array_api.empty(shape_stack, device=args[0].device)
        slices_stack = [slice(0, i) for i in shape_stack]

        for i, array in enumerate(args):
            slices_stack[self.axis] = slice(i, i + 1) # the corresponding position on the stack
            array_stack[tuple(slices_stack)] = array
        
        return array_stack
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A: Tensor):
        ### BEGIN YOUR SOLUTION
        # Author: Qingzheng Wang
        shape = list(A.shape)
        slices = [slice(0, i) for i in shape]

        shape_split = list(A.shape) # don't = shape, which will make shape_split and shape referred to same instance, dangerous!
        shape_split.pop(self.axis)
        tensor_splits = []

        for i in range(shape[self.axis]):
            slices[self.axis] = slice(i, i + 1)
            tensor_split = A[tuple(slices)].compact().reshape(shape_split)
            tensor_splits.append(tensor_split)

        return tuple(tensor_splits)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        """
        axes:       the axes to dialate, e.g. A, shape: (2, 3, 4), 
                    axes: (0, 1) means dialate axis 0 and 1 (with shape 2, 3 respectively)
        dilation:  the amount of dilation, add the dilation number of 0s between
                    the original data on the specific axis, like dilation = 2, the orignal data
                    is [1, 1, 1, 1], the dialated is [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
        """
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # Author: Qingzheng Wang
        shape_dilate = list(a.shape)
        for axis in self.axes:
            if axis < a.ndim: # ensure axis in reasonable range
                shape_dilate[axis] = (self.dilation + 1) * shape_dilate[axis]

        slice_dilate = [slice(0, shape_dilate[i], 1) for i in range(len(shape_dilate))]
        for axis in self.axes:
            if axis < a.ndim:
                slice_dilate[axis] = slice(0, shape_dilate[axis], self.dilation + 1)

        array_dilate = array_api.full(shape_dilate, 0, dtype=a.dtype, device=a.device)
        array_dilate[tuple(slice_dilate)] = a

        return array_dilate
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # Author: Qingzheng Wang
        slice_undilate = [slice(0, a.shape[i], 1) for i in range(len(a.shape))]
        for axis in self.axes:
            if axis < a.ndim:
                slice_undilate[axis] = slice(0, a.shape[axis], self.dilation + 1)
        
        return a[tuple(slice_undilate)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        """
        A: (N, H, W, C_in), suppose A is an N batch of C channel H * W images
        B: (K_1, K_2, C_in, C_out), suppose B is C_out K_1 * K_2 kernels, each with C_in channels
        In the real computation, A and B could be simply 4-D arrays, without specific meaning, 
        we define the meaning here for clarity. 

        `im2col` is a modern covolution comutation method, the idea is 
        re organize the **image to columns**.
        """
        ### BEGIN YOUR SOLUTION
        # Author: Qingzheng Wang
        if self.padding > 0:
            A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))

        N, H, W, C_in = A.shape
        N_s, H_s, W_s, C_in_s = A.strides
        K_1, K_2, C_in, C_out = B.shape

        im2col_strides = (
            N_s, 
            H_s * self.stride, 
            W_s * self.stride, 
            H_s, 
            W_s, 
            C_in_s
        )
        im2col_shape = (
            N, 
            (H - K_1) // self.stride + 1, 
            (W - K_2) // self.stride + 1, 
            K_1, 
            K_2, 
            C_in
        )
        im2col_reshape = (
            N * \
            ((H - K_1) // self.stride + 1) * \
            ((W - K_2) // self.stride + 1), 
            K_1 * K_2 * C_in
        )

        A_im2col = A.as_strided(im2col_shape, im2col_strides). \
                    compact(). \
                    reshape(im2col_reshape)
        
        B_im2col = B.compact().reshape((K_1 * K_2 * C_in, C_out)) # B must be compacted before reshape, it's a common rule of NDArray

        out = A_im2col @ B_im2col
        out = out.reshape(
            (
                N, 
                (H - K_1) // self.stride + 1, 
                (W - K_2) // self.stride + 1, 
                C_out
            ))
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Author: Qingzheng Wang
        # out_grad (N, H_out, W_out, C_out)
        A, B = node.inputs # A (N, H, W, C_in) 
        B_flip = flip(B, (0, 1))
        B_flip_trans = transpose(B_flip, (2, 3)) # (K_1, K_2, C_out, C_in)
        kernel_size = B_flip_trans.shape[0]

        if self.stride > 1:
            # TODO: I still not sure why this dilate could be right here, but the test is right, so just skip thinking. 
            out_grad = dilate(out_grad, (1, 2), self.stride - 1) # TODO: may have problems, because dilate pad 0s at the last, like [1, 0, 0, 1, 0, 0], not [1, 0, 0, 1]
        
        A_grad = conv(
            out_grad, 
            B_flip_trans,
            stride=1, 
            padding=kernel_size - 1 - self.padding if self.padding > 0 else kernel_size - 1
        )

        out_grad_trans = transpose(transpose(out_grad, (0, 1)), (1, 2)) # (H_out, W_out, N, C_out)
        B_grad = conv(
            transpose(A, (0, 3)), # (C_in, H, W, N)
            out_grad_trans,
            stride=1, 
            padding=self.padding
        ) # (C_in, K_1, K_2, C_out)
        B_grad = transpose(transpose(B_grad, (0, 1)), (1, 2)) # (K_1, K_2, C_in, C_out)

        return A_grad, B_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


# Author: Qingzheng Wang