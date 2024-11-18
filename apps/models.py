import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)

# Author: Qingzheng Wang
class ConvBN(ndl.nn.Module):
    def __init__(
        self, in_channels, out_channels, 
        kernel_size, stride=1, bias=True, 
        device=None, dtype="float32"
    ):
        super().__init__()
        self.conv = nn.Conv(
            in_channels, out_channels, kernel_size, 
            stride, bias, device, dtype
        )
        self.batch_norm = nn.BatchNorm2d(
            dim=out_channels, device=device, dtype=dtype 
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # x with shape (N, C, H, W)
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        return x

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        # Author: Qingzheng Wang
        self.convbn_0 = ConvBN(3, 16, 7, 4, device=device, dtype=dtype)
        self.convbn_1 = ConvBN(16, 32, 3, 2, device=device, dtype=dtype)
        self.convbn_2 = ConvBN(32, 32, 3, 1, device=device, dtype=dtype)
        self.convbn_3 = ConvBN(32, 32, 3, 1, device=device, dtype=dtype)
        self.convbn_4 = ConvBN(32, 64, 3, 2, device=device, dtype=dtype)
        self.convbn_5 = ConvBN(64, 128, 3, 2, device=device, dtype=dtype)
        self.convbn_6 = ConvBN(128, 128, 3, 1, device=device, dtype=dtype)
        self.convbn_7 = ConvBN(128, 128, 3, 1, device=device, dtype=dtype)
        self.flatten = nn.Flatten()
        self.linear_0 = nn.Linear(128, 128, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        self.linear_1 = nn.Linear(128, 10, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        # Author: Qingzheng Wang
        x = self.convbn_0(x)
        x = self.convbn_1(x)
        res_0 = self.convbn_2(x)
        res_0 = self.convbn_3(res_0)
        x = x + res_0
        x = self.convbn_4(x)
        x = self.convbn_5(x)
        res_1 = self.convbn_6(x)
        res_1 = self.convbn_7(res_1)
        x = x + res_1
        x = self.flatten(x)
        x = self.linear_0(x)
        x = self.relu(x)
        x = self.linear_1(x)

        return x
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        # Author: Qingzheng Wang
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embedding_size, device, dtype)
        if seq_model == 'rnn':
            self.seq_model = nn.RNN(
                embedding_size, hidden_size, num_layers, 
                bias=True, nonlinearity='tanh', device=device, dtype=dtype
            )
        elif seq_model == 'lstm':
            self.seq_model = nn.LSTM(
                embedding_size, hidden_size, num_layers, 
                bias=True, device=device, dtype=dtype
            )
        elif seq_model == 'transformer':
            self.seq_model = nn.Transformer(
                embedding_size, hidden_size, num_layers,
                device=device, dtype=dtype
            )
        # else:
        #     raise ValueError(f"unsupported seq_model: {seq_model}")
        self.linear = nn.Linear(
            hidden_size, output_size, bias=True, device=device, dtype=dtype
        )
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        # Author: Qingzheng Wang
        seq_len, bs = x.shape
        x = self.embedding(x) # (seq_len, bs, embedding_dim)
        x, h = self.seq_model(x, h) # (seq_len, bs, hidden_size)
        x = x.reshape((seq_len * bs, self.hidden_size))
        x = self.linear(x) # (seq_len * bs, output_size)
        return x, h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)
