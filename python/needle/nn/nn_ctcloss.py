from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops

from .nn_basic import Module

class CTCLoss(Module):
    def __init__(self, batch_first=False, blank=0.0, reduction="mean"):
        """
        batch_first: 
            if true, the input `logits` is (batch_size, seq_length, num_classes)
            else, the input `logits` is (seq_length, batch_size, num_classes)
        """
        super().__init__()
        self.batch_first = batch_first
        self.blank = blank
        self.reduction = reduction
    
    def forward(self, logits, target, input_lengths, target_lengths):
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
        return ops.ctc_loss(
            logits, target, input_lengths, 
            target_lengths, self.batch_first, 
            self.blank, self.reduction 
        )


    
