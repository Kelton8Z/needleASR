"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module, Sequential, Tanh, ReLU


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        ones = init.ones_like(x, device=x.device, requires_grad=False)
        return ones / (ones + ops.exp(-x))
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        # Author: Qingzheng Wang
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        boundary = 1 / (hidden_size ** 0.5)
        self.W_ih = Parameter(
            init.rand(
                input_size, hidden_size, low=-boundary, high=boundary, 
                device=device, dtype=dtype, requires_grad=True
            )
        )
        self.W_hh = Parameter(
            init.rand(
                hidden_size, hidden_size, low=-boundary, high=boundary, 
                device=device, dtype=dtype, requires_grad=True
            )
        )

        self.bias_ih = None
        self.bias_hh = None
        if bias:
            self.bias_ih = Parameter(
                init.rand(
                    hidden_size, low=-boundary, high=boundary, 
                    device=device, dtype=dtype, requires_grad=True
                )
            )
            self.bias_hh = Parameter(
                init.rand(
                    hidden_size, low=-boundary, high=boundary, 
                    device=device, dtype=dtype, requires_grad=True
                )
            )
        if nonlinearity == "tanh":
            self.nonlinearity = Tanh()
        elif nonlinearity == "relu":
            self.nonlinearity = ReLU()
        else:
            raise ValueError(f"unsupported nonlinearity: {nonlinearity}")
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        # Author: Qingzheng Wang
        if h is None:
            h = init.zeros(X.shape[0], self.hidden_size, device=X.device, dtype=X.dtype, requires_grad=True)
        
        linear = None
        if self.bias_ih and self.bias_hh is not None:
            bias_ih = self.bias_ih.reshape((1, self.hidden_size)).broadcast_to((X.shape[0], self.hidden_size))
            bias_hh = self.bias_hh.reshape((1, self.hidden_size)).broadcast_to((X.shape[0], self.hidden_size))
            linear = X @ self.W_ih + bias_ih + h @ self.W_hh + bias_hh
        else:
            linear = X @ self.W_ih + h @ self.W_hh
        
        h_prime = self.nonlinearity(linear)

        return h_prime
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        # Author: Qingzheng Wang
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn_cells = [
            RNNCell(
                input_size if i == 0 else hidden_size, 
                hidden_size, bias, 
                nonlinearity, device, dtype
            ) for i in range(num_layers)
        ]
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        # Author: Qingzheng Wang
        seq_len, batch_size, input_size = X.shape
        if h0 is None:
            h0 = init.zeros(self.num_layers, batch_size, self.hidden_size, device=X.device, dtype=X.dtype, requires_grad=True)
        
        # Since we do not implement __getitem__ and __setitem__ in Tensor, 
        # here we cannot access X[t], we must first split X into a list!
        # and this rule suit for other Tensors. 
        X_split = ops.split(X, axis=0)
        h0_split = ops.split(h0, axis=0)
        h_t = h0_split # (num_layers, batch_size, hidden_size), hiddens of time t
        h_t_final = [] # will be (seq_len, batch_size, hidden_size), hiddens at the final layer
        for t in range(seq_len):
            h_t_minus_1 = h_t
            h_t = []
            for l in range(self.num_layers):
                if l == 0:
                    h_t_l = self.rnn_cells[l](X_split[t], h_t_minus_1[l])
                else:
                    h_t_l = self.rnn_cells[l](h_t[-1], h_t_minus_1[l])
                h_t.append(h_t_l)
            h_t_final.append(h_t[-1])

        h_t_final = ops.stack(tuple(h_t_final), axis=0)
        h_n = ops.stack(tuple(h_t), axis=0)
        
        return h_t_final, h_n
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        # Author: Qingzheng Wang
        self.hidden_size = hidden_size
        boundary = 1 / (hidden_size ** 0.5)

        self.W_ih = Parameter(
            init.rand(
                input_size, 4 * hidden_size, low=-boundary, high=boundary, 
                device=device, dtype=dtype, requires_grad=True
            )
        )
        self.W_hh = Parameter(
            init.rand(
                hidden_size, 4 * hidden_size, low=-boundary, high=boundary, 
                device=device, dtype=dtype, requires_grad=True
            )
        )

        self.bias_ih = None
        self.bias_hh = None
        if bias:
            self.bias_ih = Parameter(
                init.rand(
                    4 * hidden_size, low=-boundary, high=boundary, 
                    device=device, dtype=dtype, requires_grad=True
                )
            )
            self.bias_hh = Parameter(
                init.rand(
                    4 * hidden_size, low=-boundary, high=boundary, 
                    device=device, dtype=dtype, requires_grad=True
                )
            )
        self.sigmoid_i = Sigmoid()
        self.sigmoid_f = Sigmoid()
        self.tanh_g = Tanh()
        self.sigmoid_o = Sigmoid()
        self.tanh_h = Tanh()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        # Author: Qingzheng Wang
        batch_size, input_size = X.shape
        if h is None:
            h0 = init.zeros(
                X.shape[0], self.hidden_size, device=X.device, 
                dtype=X.dtype, requires_grad=True
            )
            c0 = init.zeros(
                X.shape[0], self.hidden_size, device=X.device, 
                dtype=X.dtype, requires_grad=True
            )
        else:
            h0, c0 = h
        
        linear = None
        if self.bias_ih and self.bias_hh is not None:
            bias_ih = self.bias_ih.reshape((1, 4 * self.hidden_size)).broadcast_to((X.shape[0], 4 * self.hidden_size))
            bias_hh = self.bias_hh.reshape((1, 4 * self.hidden_size)).broadcast_to((X.shape[0], 4 * self.hidden_size))
            linear = X @ self.W_ih + bias_ih + h0 @ self.W_hh + bias_hh # (batch_size, 4 * hidden_size)
        else:
            linear = X @ self.W_ih + h0 @ self.W_hh # (batch_size, 4 * hidden_size)
        
        linear = linear.reshape((batch_size, 4, self.hidden_size))
        linear_split = ops.split(linear, axis=1) # [(batch_size, 1, self.hidden_size), (...), ...]
        i = self.sigmoid_i(linear_split[0].reshape((batch_size, self.hidden_size))) # (batch_size)
        f = self.sigmoid_f(linear_split[1].reshape((batch_size, self.hidden_size)))
        g = self.tanh_g(linear_split[2].reshape((batch_size, self.hidden_size)))
        o = self.sigmoid_o(linear_split[3].reshape((batch_size, self.hidden_size)))
        c_prime = f * c0 + i * g
        h_prime = o * self.tanh_h(c_prime)

        return h_prime, c_prime
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        # Author: Qingzheng Wang
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm_cells = [
            LSTMCell(
                input_size if i == 0 else hidden_size, 
                hidden_size, bias, 
                device, dtype
            ) for i in range(num_layers)
        ]
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        # Author: Qingzheng Wang
        seq_len, batch_size, input_size = X.shape
        if h is None:
            h0 = init.zeros(
                self.num_layers, batch_size, self.hidden_size, 
                device=X.device, dtype=X.dtype, requires_grad=True
            )
            c0 = init.zeros(
                self.num_layers, batch_size, self.hidden_size, 
                device=X.device, dtype=X.dtype, requires_grad=True
            )
        else:
            h0, c0 = h

        # Since we do not implement __getitem__ and __setitem__ in Tensor, 
        # here we cannot access X[t], we must first split X into a list!
        # and this rule suit for other Tensors. 
        X_split = ops.split(X, axis=0)
        h0_split = ops.split(h0, axis=0)
        c0_split = ops.split(c0, axis=0)

        h_t = h0_split # (num_layers, batch_size, hidden_size), hiddens of time t
        h_t_final = [] # will be (seq_len, batch_size, hidden_size), hiddens at the final layer
        c_t = c0_split # (num_layers, batch_size, hidden_size), cells of time t

        for t in range(seq_len):
            h_t_minus_1 = h_t
            c_t_minus_1 = c_t
            h_t = []
            c_t = []
            for l in range(self.num_layers):
                if l == 0:
                    h_t_l, c_t_l = self.lstm_cells[l](X_split[t], (h_t_minus_1[l], c_t_minus_1[l]))
                else:
                    h_t_l, c_t_l = self.lstm_cells[l](h_t[-1], (h_t_minus_1[l], c_t_minus_1[l]))
                h_t.append(h_t_l)
                c_t.append(c_t_l)
            h_t_final.append(h_t[-1])

        h_t_final = ops.stack(tuple(h_t_final), axis=0)
        h_n = ops.stack(tuple(h_t), axis=0)
        c_n = ops.stack(tuple(c_t), axis=0)
        
        return h_t_final, (h_n, c_n)
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(
            init.randn(
                num_embeddings, embedding_dim, mean=0.0, std=1.0, 
                device=device, dtype=dtype, requires_grad=True
            )
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, batch_size = x.shape
        x_one_hot = init.one_hot(self.num_embeddings, x, device=self.device, dtype=self.dtype) # (seq_len, bs, num_embeddings)
        x_one_hot = x_one_hot.reshape((seq_len * batch_size, self.num_embeddings)) # (seq_len * bs, num_embeddings)
        embedding = x_one_hot @ self.weight # (seq_len * bs, embedding_dim)
        embedding = embedding.reshape((seq_len, batch_size, self.embedding_dim))
        return embedding
        ### END YOUR SOLUTION

# Author: Qingzheng Wang