"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.kaiming_uniform(
            fan_in=in_features, 
            fan_out=out_features, 
            device=device, 
            dtype=dtype, 
            requires_grad=True
            )
        )
        self.bias = (
            Parameter(
            init.kaiming_uniform(
                fan_in=out_features, 
                fan_out=1, 
                device=device, 
                dtype=dtype, 
                requires_grad=True
            ).transpose()
            ) if bias else None
        )   
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # FIX: suit NDArray backend @ 2024/11/13
        # Author: Qingzheng Wang
        X_shape = X.shape
        X = X.reshape((-1, X_shape[-1]))
        x_w = X @ self.weight # (-1, out_features)

        if self.bias is not None:
            bias_broadcast = self.bias.broadcast_to(x_w.shape)
            Z = x_w + bias_broadcast
        else:
            Z = x_w

        Z = Z.reshape((*X_shape[:-1], self.out_features))
        return Z
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        shape = X.shape
        flatten_size = shape[1]
        if len(shape) > 2:
            for size in shape[2:]:
                flatten_size *= size
        return ops.reshape(X, (shape[0], flatten_size))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    """
    Get the mean softmax loss. 
    """
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        num_classes = logits.shape[-1]
        y_one_hot = init.one_hot(num_classes, y, device=logits.device, dtype=logits.dtype)
        logits_true = ops.summation(ops.multiply(y_one_hot, logits), axes=1)
        loss_softmax = ops.log(ops.summation(ops.exp(logits), axes=1)) - logits_true
        loss_softmax_mean = ops.summation(loss_softmax, axes=0) / loss_softmax.shape[0]
        return loss_softmax_mean
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim # dim is the number of channels
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.ones(dim, device=device, dtype=dtype), 
            device=device, 
            dtype=dtype
        )
        self.bias = Parameter(
            init.zeros(dim, device=device, dtype=dtype), 
            device=device, 
            dtype=dtype
        )
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        weight_broadcast = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        bias_broadcast = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        
        if self.training:
            mean = x.sum(0) / batch_size
            var = ((x - mean.reshape((1, self.dim)).broadcast_to(x.shape)) ** 2).sum(0) / batch_size
            std_eps = (var + self.eps) ** 0.5
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data
            mean_broadcast = mean.reshape((1, self.dim)).broadcast_to(x.shape)
            std_eps_broadcast = std_eps.reshape((1, self.dim)).broadcast_to(x.shape)
        else:
            mean_broadcast = self.running_mean.reshape((1, self.dim)).broadcast_to(x.shape)
            std_eps_broadcast = (self.running_var + self.eps).reshape((1, self.dim)).broadcast_to(x.shape) ** 0.5
        
        x_batch_norm = weight_broadcast * ((x - mean_broadcast) / std_eps_broadcast) + bias_broadcast
        return x_batch_norm
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2)) # nchw


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.ones(dim, device=device, dtype=dtype), 
            device=device, 
            dtype=dtype, 
            requires_grad=True
        )
        self.bias = Parameter(
            init.zeros(dim, device=device, dtype=dtype), 
            device=device, 
            dtype=dtype, 
            requires_grad=True
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # FIX: suit batch input (b, t, d), and NDArray backend @ 2024/11/13
        # Author: Qingzheng Wang
        feat_size = x.shape[-1]
        assert self.dim == feat_size, "Input feature size does not match LayerNorm dimension specification."
        ndim = len(x.shape)

        mean = x.sum(axes=ndim - 1) / feat_size
        mean = mean.reshape((*x.shape[:-1], 1)).broadcast_to(x.shape)

        var = ((x - mean) ** 2).sum(axes=ndim - 1) / feat_size
        std_eps = (var + self.eps) ** 0.5
        std_eps = std_eps.reshape((*x.shape[:-1], 1)).broadcast_to(x.shape)
        
        weight_broadcast = self.weight.reshape((1, ) * (ndim - 1) + (feat_size, )).broadcast_to(x.shape)
        bias_broadcast = self.bias.reshape((1, ) * (ndim - 1) + (feat_size, )).broadcast_to(x.shape)
        
        x_layer_norm = weight_broadcast * ((x - mean) / std_eps) + bias_broadcast

        return x_layer_norm
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            drop_mask = init.randb(*x.shape, p=1 - self.p, device=x.device, dtype=x.dtype)
            x = x * drop_mask / (1 - self.p)
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION

# Author: Qingzheng Wang