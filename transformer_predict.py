import torch
import torch.nn as nn
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
from model_transformer import TransAm

torch.manual_seed(0)
np.random.seed(0)

'''用前input_window天的数据预测后output_window天的数据'''
input_window = 20
output_window = 1
batch_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

'''训练数据train_seq和标签数据长度相同train_label'''
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]    #把数据分成长为20的一段一段的
        train_label = input_data[i + output_window:i + tw + output_window]  #比train_seq往后推output_window长度
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(inout_seq)

'''获取训练数据和测试数据[(data,target),...]'''
def get_data():
    series = pd.read_csv('./NSE-TATAGLOBAL11.csv', usecols=['Close'])
    scaler = MinMaxScaler(feature_range=(-1, 1))
    series = scaler.fit_transform(series.values.reshape(-1, 1)).reshape(-1)
    train_samples = int(0.8 * len(series))
    train_data = series[:train_samples]
    test_data = series[train_samples:]
    train_sequence = create_inout_sequences(train_data, input_window)
    train_sequence = train_sequence[:-output_window]
    test_data = create_inout_sequences(test_data, input_window)
    test_data = test_data[:-output_window]
    return train_sequence.to(device), test_data.to(device)

def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return input, target

def train(train_data):
    model.train()
    for batch_index, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        start_time = time.time()
        total_loss = 0
        '''train_data:[[[train20],[target20]],...]'''
        data, targets = get_batch(train_data, i, batch_size)
        '''data[20,64,1]'''
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()
        total_loss += loss.item()
        # log_interval = int(len(train_data) / batch_size / 5)
        # if batch_index % log_interval == 0 and batch_index > 0:
        #     cur_loss = total_loss / log_interval
        #     elapsed = time.time() - start_time
            # print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | {:5.2f} ms | loss {:5.5f} | ppl {:8.2f}'
            # .format(epoch, batch_index, len(train_data) // batch_size, scheduler.get_lr()[0], elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))

def evaluate(eval_model, data_source):
    eval_model.eval()
    total_loss = 0
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            output = eval_model(data)
            total_loss += len(data[0]) * criterion(output, targets).cpu().item()
    return total_loss / len(data_source)

def plot_and_loss(eval_model, data_source, epoch):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1)
            output = eval_model(data)
            total_loss += criterion(output, target).item()
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)
    plt.plot(test_result, color="red")
    plt.plot(truth, color="blue")
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.savefig('./graph/transformer-epoch%d.png' % epoch)
    # 
    # plt.show()
    plt.close()
    return total_loss / i

if not os.path.exists('./graph'):
    os.mkdir('./graph')
train_data, val_data = get_data()
model = TransAm().to(device)
criterion = nn.MSELoss()
lr = 0.0001
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
epochs = 200

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data)

    if (epoch % 10 == 0):
        val_loss = plot_and_loss(model, val_data, epoch)
    else:
        val_loss = evaluate(model, val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
    print('-' * 89)
    # scheduler.step()

