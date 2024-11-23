from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops

from .nn_basic import Module

class CTCLoss(Module):
    def __init__(self, blank=0.0, reduction="mean"):
        super().__init__()
        self.blank = blank
        self.reduction = reduction
    
    def forward(self, logits, target, input_lengths, target_lengths):
        return ops.ctc_loss(
            logits, target, input_lengths, 
            target_lengths, self.blank, self.reduction 
        )


    
