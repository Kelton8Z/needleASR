from typing import List
from needle.autograd import Tensor
import needle.backend_ndarray.ndarray as ndarray
from needle import ops
import needle.init as init
import numpy as np
from .nn_sequence import Embedding
from .nn_basic import (
    Parameter, 
    Module, 
    ReLU,
    Dropout,
    LayerNorm1d,
    Linear,
    Sequential
)


class MultiHeadAttention(Module):
    """
    The multi-head self attention module.
    """
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        self.causal = causal
        self.dropout = Dropout(dropout)

    def create_causal_mask(self, i, j, device):
        """
        return a triangular causal mask with shape (1, 1, i, j).
        """
        mask = -np.finfo(np.float32).max * np.triu(
            np.ones((1, 1, i, j), dtype=np.float32), j - i + 1)
        # result a mask like this if i = 3, j = 3:
        # [
        #   [0,     -inf,   -inf], 
        #   [0,     0,      -inf], 
        #   [0,     0,      0   ] 
        # ]

        return ndarray.array(
            mask, device=device)

    def matmul(self, a, b_transpose):
        """
        batched matrix multiplication;
        a: (..., T, D)
        b_transpose: (..., T, D)
        result: (..., T, T)
        Note: 
            Althogh b_transpose is called "transpose", it is a not transposed
            transposed matrix of b. We just want to compute a @ b.T, but since
            we did not implement batched matrix multiplication in NDArray backend, 
            we have to use this trick. 

            if want to a @ b.T, please input a, b
            if want to a @ b, please input a, b.T
        Explain: 
            This batched matrix multiplication trick broadcast a from (..., T, D)
            to (..., T, `T`, D), and broadcast b_transpose from (..., T, D) to 
            (..., `T`, T, D). `T` in a is the broadcasted axis, which means the 
            following D-size (the D in the last axis) vectors are same. Simalarly, `T`
            in b_transpose is the broadcasted axis, which means the following T * D size
            arrays are same. Suppose T = 2, D = 3, the original a and b_transpose are:
            [
                [1, 2, 3], 
                [4, 5, 6]
            ]
            [
                [a, b, c], 
                [d, e, f]
            ]
            Then the broadcasted a and b_transpose are:
            [
                    [[1, 2, 3], [1, 2, 3]], 
                    [[4, 5, 6], [4, 5, 6]]
            ]
            [
                    [[a, b, c], [d, e, f]], 
                    [[a, b, c], [d, e, f]]
            ]
            Then we can multiply them element-wise and sum them 
            along the last axis, the result is:
            [
                [a + 2b + 3c, d + 2e + 3f], 
                [4a + 5b + 6c, 4d + 5e + 6f]
            ]
            which is exactly the result of a @ b.T!!!
        """
        a_shape = (*a.shape[:-1], 1, *a.shape[-1:])
        a = a.reshape(a_shape)

        b_transpose_shape = (*b_transpose.shape[:-2], 1, *b_transpose.shape[-2:])
        b_transpose = b_transpose.reshape(b_transpose_shape)

        broadcast_shape = list(a_shape)
        broadcast_shape[-2] = b_transpose_shape[-2]
        a = a.broadcast_to(broadcast_shape)

        broadcast_shape = list(b_transpose_shape)
        broadcast_shape[-3] = a_shape[-3]
        b_transpose = b_transpose.broadcast_to(broadcast_shape)

        return (a * b_transpose).sum(len(a.shape) - 1)

    def softmax(self, logit):
        """
        The softmax function; 
        """
        max_val = Tensor(
            logit.realize_cached_data().max(axis=3),
            device=logit.device,
            dtype=logit.dtype,
            requires_grad=False
        )

        max_val = max_val.reshape((*logit.shape[:-1], 1))
        max_val = max_val.broadcast_to(logit.shape)

        probs = ops.exp(logit - max_val)

        denom = probs.sum(axes=3)
        denom = denom.reshape((*logit.shape[:-1], 1))
        denom = denom.broadcast_to(logit.shape)

        return probs / denom

    def forward(
        self,
        q, k, v,
    ):
        """
        The forward function of the MultiHeadAttention activation function.
        Input: three states q, k, v, with shape (batch_size, num_head, seq_len, dim_head)
        Output: the activation output `result` and attention softmax probability `probs` (with dropout applied)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, keys_values_len, k_dim = k.shape
        _, _, _, v_dim = v.shape

        assert q_dim == k_dim == v_dim

        result = None
        probs = None

        ### BEGIN YOUR SOLUTION
        att = self.matmul(q, k) # (batch_size, num_head, queries_len, keys_values_len)
        att /= np.sqrt(q_dim)

        if self.causal:
            mask = self.create_causal_mask(queries_len, keys_values_len, self.device) # (1, 1, queries_len, keys_values_len)
            mask = mask.broadcast_to(att.shape)
            att += mask
        
        probs = self.softmax(att)
        probs = self.dropout(probs)

        result = self.matmul(probs, v.transpose((2, 3))) # (batch_size, num_head, queries_len, v_dim)
        ### END YOUR SOLUTION
        return result, probs


class AttentionLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        *,
        k_features: int = None,
        v_features: int = None,
        out_features: int = None,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        if k_features is None:
            k_features = q_features
        if v_features is None:
            v_features = q_features
        if out_features is None:
            out_features = q_features

        self.q_features = q_features
        self.k_features = k_features
        self.v_features = v_features
        self.out_features = out_features

        self.num_head = num_head
        self.dim_head = dim_head

        self.prenorm_q = LayerNorm1d(
            q_features, device=device, dtype=dtype)
        self.prenorm_k = LayerNorm1d(
            k_features, device=device, dtype=dtype)
        self.prenorm_v = LayerNorm1d(
            v_features, device=device, dtype=dtype)

        inner_dim = num_head * dim_head
        
        self.q_projection = Linear(
            q_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.k_projection = Linear(
            k_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.v_projection = Linear(
            v_features, inner_dim, bias=False,
            device=device, dtype=dtype)

        self.attn = MultiHeadAttention(
            dropout=dropout, causal=causal,
            device=device, dtype=dtype)

        self.out_projection = Linear(
            inner_dim, out_features, bias=False,
            device=device, dtype=dtype)

    def forward(
        self,
        q, k=None, v=None,
    ):
        """
        The forward function of the self-attention layer.
        If k, v not None, cross attention.
        Input: `q` with shape (batch_size, q_len, q_dim)
               `k` (if not None) with shape (batch_size, kv_len, k_dim)
               `v` (if not None) with shape (batch_size, kv_len, v_dim)
        Output: the output `result` with shape (batch_size, kv_len, out_features)
        """

        if k is None:
            k = q
        if v is None:
            v = q

        batch_size, queries_len, q_dim = q.shape
        _, kv_len, k_dim = k.shape
        _, _, v_dim = v.shape

        result = None

        ### BEGIN YOUR SOLUTION
        q = self.q_projection(self.prenorm_q(q))
        k = self.k_projection(self.prenorm_k(k))
        v = self.v_projection(self.prenorm_v(v))

        # (batch_size, num_head, q_len, dim_head)
        q = q.reshape((batch_size, queries_len, self.num_head, self.dim_head)).transpose((1, 2))
        k = k.reshape((batch_size, kv_len, self.num_head, self.dim_head)).transpose((1, 2))
        v = v.reshape((batch_size, kv_len, self.num_head, self.dim_head)).transpose((1, 2))

        x, probs = self.attn(q, k, v)
        x = x.transpose((1, 2)).reshape((batch_size, queries_len, self.num_head * self.dim_head))
        result = self.out_projection(x)
        ### END YOUR SOLUTION

        return result


class TransformerLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        hidden_size: int,
        *,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        ### BEGIN YOUR SOLUTION
        self.attn = AttentionLayer(
            q_features, num_head, dim_head, 
            out_features=q_features, dropout=dropout, 
            causal=causal, device=device, dtype=dtype
        )
        self.dropout_0 = Dropout(dropout)

        self.layer_norm = LayerNorm1d(q_features, device=device, dtype=dtype)
        self.linear_1 = Linear(q_features, hidden_size, device=device, dtype=dtype)
        self.relu_1 = ReLU()
        self.dropout_1 = Dropout(dropout)

        self.linear_2 = Linear(hidden_size, q_features, device=device, dtype=dtype)
        self.dropout_2 = Dropout(dropout)
        ### END YOUR SOLUTION

    def forward(
        self,
        x
    ):
        """
        The forward function of a Transformer Layer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """

        batch_size, seq_len, x_dim = x.shape

        ### BEGIN YOUR SOLUTION
        res = self.dropout_0(self.attn(x))
        x = x + res

        res = self.dropout_2(self.linear_2(self.dropout_1(self.relu_1(self.linear_1(self.layer_norm(x))))))
        x = x + res
        ### END YOUR SOLUTION

        return x


class Transformer(Module):

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int, 
        *,
        num_head: int = 8,
        dim_head: int = 32,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        batch_first = False,
        sequence_len = 2048, 
        if_positional_embedding = False
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.batch_first = batch_first
        self.sequence_len = sequence_len

        ### BEGIN YOUR SOLUTION
        # Author: Qingzheng Wang
        self.if_positional_embedding = if_positional_embedding
        if if_positional_embedding:
            self.positional_encoder = Embedding(
                num_embeddings=sequence_len, embedding_dim=embedding_size, 
                device=self.device, dtype=self.dtype
            )
            
        self.transformer = Sequential(*[
            TransformerLayer(
                embedding_size, num_head, dim_head, hidden_size, 
                dropout=dropout, causal=causal, device=device, 
                dtype=dtype
            ) for _ in range(num_layers)
        ])
        
        ### END YOUR SOLUTION

    def forward(
        self,
        x, h=None
    ):

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        ### BEGIN YOUR SOLUTION
        # Author: Qingzheng Wang
        batch_size, seq_len, input_size = x.shape

        time_ids = np.arange(seq_len).reshape(-1, 1)
        time_ids = ndarray.array(time_ids, device=self.device).broadcast_to((seq_len, batch_size))
        time_ids = Tensor(time_ids, device=self.device, dtype=self.dtype)

        if self.if_positional_embedding:
            positional_embedding = self.positional_encoder(time_ids) # (seq_len, batch_size, embedding_size)
            positional_embedding = positional_embedding.transpose((0, 1))
            x = x + positional_embedding
        
        x = self.transformer(x)
        ### END YOUR SOLUTION

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        return x, init.zeros_like(x)
# Author: Qinzheng Wang