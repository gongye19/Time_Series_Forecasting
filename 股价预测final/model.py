import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.args = args
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.num_layers = num_layers
        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Readout layer 在LSTM后再加一个全连接层，因为是回归问题，所以不能在线性层后加激活函数
        self.fc1 = nn.Linear(args.back_days*hidden_dim, (args.back_days*hidden_dim)//2)
        self.tanh= nn.Tanh()
        self.fc2 = nn.Linear((args.back_days*hidden_dim)//2, output_dim)

    def forward(self, x):
        # print(x.shape)
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # 这里x.size(0)就是batch_size
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # print(out.shape)
        # out, (hn, cn) = self.lstm(x, (h0.detch(), c0.detch()))
        out = out.reshape(-1,self.args.back_days*self.hidden_dim)
        out = self.fc2(self.tanh(self.fc1(out)))
        
        # print(out.shape)
        # m=9/0
        return out