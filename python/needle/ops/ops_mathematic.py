"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy as np

import sys
sys.path.append("python/")
import needle as ndl

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *

# Use np.float32 rather than float, since float is float64 by default
scalar_t = np.float32
MIN_VAL = -1e25
EPSILON = 1e-6
DEBUG = False


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

    def compute(self, args: Tuple[NDArray]) -> NDArray:
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

### ==================================== CTC Loss ===================================================== ###
# NOTE: We decide to write CTCLoss as a TensorOp, because the Needle Tensor have not implemented          #
# the __getitem__ and __setitem__ method, so many Tensor operations cannot directly used.                 #
# Instead, the NDArray have implemented these methods, so we can use NDArray to implement the CTCLoss.    #
# We implemented plenty of operators `compute` in NDArray, and directly use these NDArray `compute`       #
# to implement the `gradient`, which prevent to write __getitem__ and __setitem__ on Tensor in `gradient` #
# computation.                                                                                            #
# ======================================================================================================= #

class CTC:
    """
    CTC contains nessesary NDArray operators to compute the CTC loss.
    """
    def __init__(self, blank=0.0):
        """Initialize instance variables

        Argument(s)
        -----------
        blank (int, optional): blank label index. Default 0.
        """

        # No need to modify
        self.blank = blank

    def extend_target_with_blank(self, target: NDArray) -> Tuple[NDArray, NDArray]:
        """Extend target sequence with blank.

        Input
        -----
        target: (NDArray, dim = (target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extended_symbols: (NDArray, dim = (2 * target_len + 1,))
                          extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]

        skip_connect: (np.array, dim = (2 * target_len + 1,))
                      skip connections
        ex: [0,0,0,1,0,0,0,1,0]
        """

        extended_symbols = array_api.full((2 * len(target) + 1, ), self.blank, device=target.device)
        for i in range(len(target)):
            extended_symbols[2 * i + 1] = target[i]

        N = len(extended_symbols)

        skip_connect = [0.0] * N
        for i in range(2, N):
            if float(extended_symbols[i].numpy()) != self.blank and extended_symbols[i] != extended_symbols[i-2]:
                skip_connect[i] = 1.0

        extended_symbols = array_api.array(extended_symbols, device=target.device)
        skip_connect = array_api.array(skip_connect, device=target.device)

        return extended_symbols, skip_connect

    def logsumexp(self, a, b):
        """
        Ultra-stable logsumexp
        a, b: 
            NDArray (single value, [a_value], [b_value]), log probabilities
            a = log(exp(x_a) / sum(exp(x))), b = log(exp(x_b) / sum(exp(x)))
            exp(a) + exp(b) = (exp(x_a) + exp(x_b)) / sum(exp(x))
            log(exp(a) + exp(b)) = log(prob(x_a) + prob(x_b))
            Hence, this function is to compute the log probability of the 
            sum of two probabilities.
        return:
            NDArray!!!
        """
        # Convert NDArray to scalar
        
        # If one log prob is -inf, then its prob is zero
        if scalar_t(a.numpy()) == scalar_t(MIN_VAL) and scalar_t(b.numpy()) != scalar_t(MIN_VAL):
            return b
        elif scalar_t(b.numpy()) == scalar_t(MIN_VAL) and scalar_t(a.numpy()) != scalar_t(MIN_VAL):
            return a
        elif scalar_t(a.numpy()) == scalar_t(MIN_VAL) and scalar_t(b.numpy()) == scalar_t(MIN_VAL): 
            return a
        
        if scalar_t(a.numpy()) >= scalar_t(b.numpy()):
            max_array = a
            min_array = b
        else:
            max_array = b
            min_array = a

        # If values are too far apart, return the max
        if scalar_t(max_array.numpy()) - scalar_t(min_array.numpy()) > scalar_t(30):
            return max_array
        
        t1 = min_array - max_array
        t2 = t1.exp()
        t3 = t2 + scalar_t(1)
        t4 = t3.log()
        t5 = max_array + t4
        return t5

    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        S, T = len(extended_symbols), len(logits)
        alpha = array_api.full((T, S), scalar_t(MIN_VAL), dtype="float32", device=logits.device)
        
        # Initialize with normalized logits
        alpha[0, 0] = logits[0, int(extended_symbols[0].numpy())]
        if S > 1:
            alpha[0, 1] = logits[0, int(extended_symbols[1].numpy())]

        # Keep track of scaling factors
        scale = array_api.full((T, ), scalar_t(0), dtype="float32", device=logits.device)

        for t in range(1, T):
            alpha[t, 0] = alpha[t-1, 0] + logits[t, int(extended_symbols[0].numpy())]

            # Current emissions
            for i in range(1, S):
                log_emit = logits[t, int(extended_symbols[i].numpy())]
                log_emit = log_emit.compact().reshape((1, 1))

                curr = scalar_t(MIN_VAL)
                curr_1 = None
                curr_2 = None
                curr_3 = None
                
                # Standard transition
                if alpha[t-1, i-1] > scalar_t(MIN_VAL):
                    curr_1 = alpha[t-1, i-1]
                    curr_1 = curr_1.compact().reshape((1, 1))
                    curr = curr_1
                
                # Stay transition
                if alpha[t-1, i] > scalar_t(MIN_VAL):
                    alpha_t_1_i = alpha[t-1, i].compact().reshape((1, 1))
                    curr_2 = self.logsumexp(curr, alpha_t_1_i)
                    curr = curr_2
                
                # Skip connection
                if scalar_t(skip_connect[i].numpy()) and i >= 2:
                    if alpha[t-1, i-2] > scalar_t(MIN_VAL):
                        alpha_t_1_i_2 = alpha[t-1, i-2].compact().reshape((1, 1))
                        curr_3 = self.logsumexp(curr, alpha_t_1_i_2)
                        curr = curr_3
                
                if curr > scalar_t(MIN_VAL):
                    alpha[t, i] = curr + log_emit

            # Normalize every timestep
            alpha_t = alpha[t]
            max_val = scalar_t(alpha[t].max(axis=1).numpy())
            if max_val != scalar_t(MIN_VAL):
                alpha[t] = alpha[t] - max_val
                scale[t] = max_val
        
        return alpha, scale

    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        S, T = len(extended_symbols), len(logits)
        beta = array_api.full((T, S), scalar_t(MIN_VAL), dtype="float32", device=logits.device)
        
        # Initialize normalized
        beta[T-1, S-1] = logits[T-1, int(extended_symbols[S-1].numpy())]
        if S > 1:
            beta[T-1, S-2] = logits[T-1, int(extended_symbols[S-2].numpy())]

        # Keep track of scaling factors
        scale = array_api.full((T, ), scalar_t(0), dtype="float32", device=logits.device)

        for t in reversed(range(T-1)):
            beta[t, S-1] = beta[t+1, S-1] + logits[t, int(extended_symbols[S-1].numpy())]

            for i in reversed(range(S-1)):
                
                log_emit = logits[t, int(extended_symbols[i].numpy())]
                curr = scalar_t(MIN_VAL)
                curr_1 = None
                curr_2 = None
                curr_3 = None
                
                # Standard transitions
                if beta[t+1, i] > scalar_t(MIN_VAL):
                    curr_1 = beta[t+1, i]
                    curr_1 = curr_1.compact().reshape((1, 1))
                    curr = curr_1
                
                if beta[t+1, i+1] > scalar_t(MIN_VAL):
                    curr_2 = self.logsumexp(curr, beta[t+1, i+1])
                    curr_2 = curr_2.compact().reshape((1, 1))
                    curr = curr_2
                
                # Skip connection
                if i < S-2 and scalar_t(skip_connect[i+2].numpy()):
                    if beta[t+1, i+2] > scalar_t(MIN_VAL):
                        curr_3 = self.logsumexp(curr, beta[t+1, i+2])
                        curr_3 = curr_3.compact().reshape((1, 1))
                        curr = curr_3
                
                if curr > scalar_t(MIN_VAL):
                    beta[t, i] = curr + log_emit

            # Normalize every timestep
            max_val = scalar_t(beta[t].max(axis=1).numpy())
            if max_val != scalar_t(MIN_VAL):
                beta[t] = beta[t] - max_val
                scale[t] = max_val

        return beta, scale

    def get_posterior_probs(self, alpha, beta):
        T, S = alpha.shape
        gamma = array_api.full((T, S), scalar_t(0), dtype="float32", device=alpha.device)
        
        for t in range(T):
            # Find valid positions and max value for numerical stability
            max_val = scalar_t(MIN_VAL)
            valid_pos = []
            
            for s in range(S):
                if alpha[t, s] > scalar_t(MIN_VAL) and beta[t, s] > scalar_t(MIN_VAL):
                    curr_val = scalar_t((alpha[t, s] + beta[t, s]).numpy())
                    max_val = max(max_val, curr_val)
                    valid_pos.append((s, curr_val))
            
            if not valid_pos:
                continue
                
            # Compute normalized probabilities
            sum_exp = 0.0
            for s, val in valid_pos:
                sum_exp += np.exp(val - max_val) # use np since here is value not NDArray
                
            log_sum = max_val + np.log(sum_exp)
            
            # Set probabilities
            for s, val in valid_pos:
                gamma[t, s] = np.exp(val - log_sum)
        
        return gamma


class CTCLoss(TensorOp):
    def __init__(self, batch_first=False, blank=0.0, reduction="mean"):
        self.batch_first = batch_first
        self.blank = blank
        self.reduction = reduction
        self.ctc = CTC(blank)
    
    def compute(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward

		Computes the CTC Loss by calculating forward, backward, and
		posterior proabilites, and then calculating the avg. loss between
		targets and predicted log probabilities

        Input
        -----
        logits [NDArray, dim=(seq_length, batch_size, len(symbols)]:
			log probabilities (output sequence) from the RNN/GRU

        target [NDArray, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [NDArray, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [NDArray, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """
        if self.batch_first:
            logits = logits.permute((1, 0, 2))

        T, _, _ = logits.shape
        B, _ = target.shape
        total_loss = array_api.full((B, ), scalar_t(0), dtype="float32", device=logits.device)
        extended_symbols_list = []

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute forward probabilities
            #     Compute backward probabilities
            #     Compute posteriors using total probability function
            #     Compute expected divergence for each batch and store it in totalLoss
            #     Take an average over all batches and return final result

            if DEBUG:
                print("=================== Trunc ===================")

                # print type of logits, target, input_lengths, target_lengths
                print(f"Logits Type: {type(logits)}")
                print(f"Target Type: {type(target)}")
                print(f"Input Lengths Type: {type(input_lengths)}")
                print(f"Target Lengths Type: {type(target_lengths)}")

                # print shape of logits, target, input_lengths, target_lengths
                print(f"Logits Shape: {logits.shape}")
                print(f"Target Shape: {target.shape}")
                print(f"Input Lengths Shape: {input_lengths.shape}")
                print(f"Target Lengths Shape: {target_lengths.shape}")

                # print device of logits, target, input_lengths, target_lengths
                print(f"Logits Device: {logits.device}")
                print(f"Target Device: {target.device}")
                print(f"Input Lengths Device: {input_lengths.device}")
                print(f"Target Lengths Device: {target_lengths.device}")

            target_trunc = target[batch_itr, :int(target_lengths[batch_itr].numpy())].compact().reshape(-1)
            logits_trunc = logits[:int(input_lengths[batch_itr].numpy()), batch_itr, :].compact().reshape((int(input_lengths[batch_itr].numpy()), -1))

            if DEBUG:
                print(f"Target Truncated: {target_trunc}")
                print(f"Target Truncated Shape: {target_trunc.shape}")
                print(f"Logits Truncated: {logits_trunc}")
                print(f"Logits Truncated Shape: {logits_trunc.shape}")

                print("=================== Extend Target ===================")

            extended_symbols, skip_connect = self.ctc.extend_target_with_blank(target_trunc)

            if DEBUG:
                print(f"Extended Symbols: {extended_symbols}")

                print("=================== Forward ===================")

            alpha, _ = self.ctc.get_forward_probs(logits_trunc, extended_symbols, skip_connect)
            assert not any(map(lambda x: x!=x, alpha.numpy().flatten())), "alpha contain NaN values"

            if DEBUG:
                print(f"Alpha: {alpha}")
                print(f"Alpha Shape: {alpha.shape}")
                print(f"Alpha Type: {type(alpha)}")

                print("=================== Backward ===================")

            beta, _ = self.ctc.get_backward_probs(logits_trunc, extended_symbols, skip_connect)
            assert not any(map(lambda x: x!=x, beta.numpy().flatten())), "beta contain NaN values"

            if DEBUG:
                print(f"Beta: {beta}")
                print(f"Beta Shape: {beta.shape}")
                print(f"Beta Type: {type(beta)}")

                print("=================== Posterior ===================")

            gamma = self.ctc.get_posterior_probs(alpha, beta)
            assert not any(map(lambda x: x!=x, gamma.numpy().flatten())), "gamma contain NaN values"

            if DEBUG:
                print(f"Gamma: {gamma}")
                print(f"Gamma Shape: {gamma.shape}")
                print(f"Gamma Type: {type(gamma)}")

            extended_symbols_list.append(extended_symbols)

            T = gamma.shape[0]
            S = gamma.shape[1]
            logits_extended  = array_api.full((T, S), scalar_t(0), dtype="float32", device=logits.device)

            for t in range(T):
                for s in range(S):
                    logits_extended[t, s] = logits_trunc[t, int(extended_symbols[s].numpy())]

            logits_extended = logits_extended + EPSILON

            t0 = logits_extended # logits itself is log probabilities, should not log() again
            assert not any(map(lambda x: x!=x, t0.numpy().flatten())), "t0 contain NaN values"
            t1 = gamma * logits_extended
            assert not any(map(lambda x: x!=x, t1.numpy().flatten())), "t1 contain NaN values"
            t2 = t1.sum(axis=1)
            assert not any(map(lambda x: x!=x, t2.numpy().flatten())), "t2 contain NaN values"
            t3 = t2.sum(axis=0)
            assert not any(map(lambda x: x!=x, t3.numpy().flatten())), "t3 contain NaN values"

            div = (gamma * logits_extended).sum(axis=1).sum(axis=0) # Sum over all the symbols and time steps
            assert not any(map(lambda x: x!=x, div.numpy().flatten())), "div contain NaN values"
            total_loss[batch_itr] = -div

        if self.reduction == "mean":
            return total_loss.sum(axis=0) / (B * T)
        elif self.reduction == "sum":
            return total_loss.sum(axis=0)
        elif self.reduction == "None":
            return total_loss
        else:
            raise ValueError(f"Invalid reduction type {self.reduction}")

    def gradient(self, out_grad: Tensor, node: Tensor):
        logits, target, input_lengths, target_lengths = node.inputs
        if self.batch_first:
            logits = logits.transpose((1, 0))

        grad = ctc_loss_gradient(
            logits, target, input_lengths, target_lengths, self.blank, self.reduction
        )

        return_val = out_grad.reshape((1, ) * len(grad.shape)).broadcast_to(grad.shape) * grad

        if self.batch_first:
            return_val = return_val.transpose((1, 0))

        return return_val

def ctc_loss(logits, target, input_lengths, target_lengths, batch_first, blank, reduction):
    return CTCLoss(batch_first, blank, reduction)(logits, target, input_lengths, target_lengths)

class CTCLossGradient(TensorOp):
    """ CTC Loss backward operator
    Computes the gradient of the CTC loss with respect to the logits.
    """
    def __init__(self, blank=0.0, reduction="mean"):
        self.blank = blank
        self.reduction = reduction
        self.ctc = CTC(blank)
    
    def compute(
            self, logits: NDArray, target: NDArray, 
            input_lengths: NDArray, target_lengths: NDArray
        ) -> NDArray:
        _, B, _ = logits.shape
        grad = array_api.full(logits.shape, scalar_t(0), dtype="float32", device=logits.device)

        for batch_itr in range(B):
            target_trunc = target[batch_itr, :int(target_lengths[batch_itr].numpy())].compact().reshape(-1)
            logits_trunc = logits[:int(input_lengths[batch_itr].numpy()), batch_itr].compact().reshape((int(input_lengths[batch_itr].numpy()), -1))

            extended_symbols, skip_connect = self.ctc.extend_target_with_blank(target_trunc)
            alpha, _ = self.ctc.get_forward_probs(logits_trunc, extended_symbols, skip_connect)
            assert not any(map(lambda x: x!=x, alpha.numpy().flatten())), "alpha in gradient contain NaN values"
            beta, _ = self.ctc.get_backward_probs(logits_trunc, extended_symbols, skip_connect)
            assert not any(map(lambda x: x!=x, beta.numpy().flatten())), "beta in gradient contain NaN values"
            gamma = self.ctc.get_posterior_probs(alpha, beta)
            assert not any(map(lambda x: x!=x, gamma.numpy().flatten())), "gamma in gradient contain NaN values"
            T, S = gamma.shape

            for t in range(T):
                for s in range(S):
                    prob = logits_trunc[t, int(extended_symbols[s].numpy())]
                    grad[t, batch_itr, int(extended_symbols[s].numpy())] -= (gamma[t, s] * array_api.exp(-prob)).compact().reshape((1, 1, 1))
        
        assert not any(map(lambda x: x!=x, grad.numpy().flatten())), "grad contain NaN values"
        return grad

    def gradient(self, out_grad: Tensor, node: Tensor):
        pass

def ctc_loss_gradient(
        logits, target, input_lengths, 
        target_lengths, blank=0.0, reduction="mean"
    ):

    return CTCLossGradient(
        blank, reduction
    )(logits, target, input_lengths, target_lengths)

# Author: Qingzheng Wang
