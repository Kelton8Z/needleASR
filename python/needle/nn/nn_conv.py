"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        # Author: Qingzheng Wang
        fan_in = kernel_size * kernel_size * in_channels
        fan_out = kernel_size * kernel_size * out_channels
        kernel_shape = (kernel_size, kernel_size, in_channels, out_channels)

        self.weight = Parameter(
            init.kaiming_uniform(
            fan_in, 
            fan_out, 
            shape=kernel_shape, 
            device=device, 
            dtype=dtype, 
            requires_grad=True
            )
        )

        self.bias = None
        if bias:
            boundary = 1.0 / (fan_in ** 0.5)
            self.bias = Parameter(
                init.rand(
                    out_channels, low=-boundary, high=boundary, 
                    device=device, dtype=dtype, requires_grad=True
                )
            )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Author: Qingzheng Wang
        # Input x must be in (B, C, H, W)
        x = x.transpose((1, 2)).transpose((2, 3))
        
        padding = self.kernel_size // 2
        conv_result = ops.conv(x, self.weight, self.stride, padding)

        if self.bias is not None:
            # Cannot assign the ops result to self.bias here,
            # I guess the reason is prevent the computational graph circulted. 
            bias = self.bias.reshape((1, 1, 1, self.out_channels))
            bias = bias.broadcast_to(conv_result.shape)
            conv_result = conv_result + bias
        
        conv_result = conv_result.transpose((2, 3)).transpose((1, 2))

        return conv_result
        ### END YOUR SOLUTION

# Author: Qingzheng Wang