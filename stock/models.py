import torch
from torch import nn

# # Define LSTM Neural Networks
# class RnnLSTM(nn.Module):
#     """
#         Parameters：
#         - input_size: feature size
#         - hidden_size: number of hidden units
#         - output_size: number of output
#         - num_layers: layers of LSTM to stack
#     """
#     @staticmethod
#     def weight_init(m):
#         if isinstance(m,nn.Linear):
#             nn.init.xavier_normal_(m.weight)
#             nn.init.constant_(m.bias,0)
#         elif isinstance(m,nn.BatchNorm1d):
#             nn.init.constant_(m.weight,1)
#             nn.init.constant_(m.bias,0)

#     def __init__(self, input_size = 1, hidden_size=5, output_size=1, num_layers=4,batch_first=True):
#         super(RnnLSTM,self).__init__()
#         # self.args = args
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
#         for name, param in self.lstm.named_parameters():
#             nn.init.uniform_(param,-0.1,0.1)
#         self.linear1 = nn.Linear(hidden_size, 12) # 全连接层
#         self.linear2 = nn.Linear(12, output_size)
#         self.con1d = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=60)
#         self.relu= nn.ReLU(inplace=True)
#         self.tanh= nn.Tanh()
#         self.apply(self.weight_init)
        
#     def forward(self, _x):
#         print(_x)
        
#         # h0 = torch.zeros(_x.size(0),self.num_layers, self.hidden_size).requires_grad_().to('cuda:0')
#         # c0 = torch.zeros( _x.size(0),self.num_layers, self.hidden_size).requires_grad_().to('cuda:0')
#         x, (hn, cn) = self.lstm(_x)
#         # x, (hn, cn) = self.lstm(_x,(h0, c0))  # _x is input, size (seq_len, batch, input_size)
#         # print(x.shape)
#         # m=8/0
#         batch, seq_len, hidden_size = x.shape  # x is output, size (seq_len, batch, hidden_size)
#         x = x.view(batch * seq_len, hidden_size)
#         x = self.linear1(x)
#         x = self.relu(x)
#         x = self.linear2(x)
#         x = x.view(seq_len, batch, -1)
#         x = x.view(1, seq_len, batch)
#         x = self.con1d(x)
#         x = torch.squeeze(x,0)

#         return x



# class RegLSTM(nn.Module):
#     def __init__(self):
#         super(RegLSTM, self).__init__()
#         # 定义LSTM
#         self.rnn = nn.LSTM(input_size, hidden_size, hidden_num_layers)
#         # 定义回归层网络，输入的特征维度等于LSTM的输出，输出维度为1
#         self.reg = nn.Sequential(
#             nn.Linear(hidden_size, 1)
#         )

#     def forward(self, x):
#         x, (ht,ct) = self.rnn(x)
#         seq_len, batch_size, hidden_size= x.shape
#         x = x.view(-1, hidden_size)
#         x = self.reg(x)
#         x = x.view(seq_len, batch_size, -1)
#         return x


class LSTM(nn.Module):
    """
        Parameters:
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, args, input_size=1, hidden_size=50, num_layers=2):
        super(LSTM,self).__init__()
        self.args = args
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 # 单向LSTM
        self.batch_size = args.batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, 1)
        self.con1d = nn.Conv1d(in_channels=5,out_channels=1,kernel_size=1)
    '''
    input(seq_len, batch, input_size)
    seq_len:序列长度,在NLP中就是句子长度,一般都会用pad_sequence补齐长度
    batch:每次喂给网络的数据条数,在NLP中就是一次喂给网络多少个句子
    input_size:特征维度,和前面定义网络结构的input_size一致。
    如果batch_fisrt=true,batch和seq_len就调换顺序
    '''
    def forward(self, input_seq):
        # print(input_seq)
        # h_0 = torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to('cuda:0')
        # c_0 = torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to('cuda:0')
        # output, (ht,ct) = self.lstm(input_seq, (h_0, c_0))
        output, (ht,ct) = self.lstm(input_seq)
        # print(output)
        pred = self.linear(output)
        pred = self.con1d(pred)
        pred = pred.view(1)

        # pred = pred[:, -1, :]
        return pred
