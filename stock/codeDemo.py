import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from torch.utils.data import DataLoader
from GetData import GetDataset
from models import LSTM
import os

def parse_args():
    parser = argparse.ArgumentParser()
    # MODIFIABLE MEVF HYPER-PARAMETERS--------------------------------------------------------------------------------
    # Model loading/saving
    parser.add_argument('--input_data', type=str, default='./NSE-TATAGLOBAL11.csv')
    parser.add_argument('--save_path', type=str, default='./saved_model')
    parser.add_argument('--checkpoint', type=str, default='./saved_model/model_1.pth')
    parser.add_argument('--seed', type=int, default=88)
    parser.add_argument('--gpu', type=str, default=0)
    parser.add_argument('--back_days', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--train', action='store_true', default=False)

    args = parser.parse_args()
    return args

def train(args, train_loader,criterion,optimizer,new_data,device):
    p = []
    # print(model)
    # m=9/0
    for i in range(args.epoch):
        for j,(input, target_train) in enumerate(train_loader):

            input, target_train = torch.tensor(input,dtype=torch.float).to(device), torch.tensor(target_train,dtype=torch.float).to(device)
            # input_data.requires_grad_()
            predict = model.forward(input)
            p.append([predict.item()])
            loss = criterion(predict,target_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), args.save_path+"/model_{}.pth".format((i+1)))
        # print(model.state_dict())
        # torch.save(model, args.save_path+"/model_{}.pth".format((i+1)))
    train = new_data[:int(len(new_data)*0.8)]
    # train = new_data
    valid = new_data[int(len(new_data)*0.8):]

    a = train[:args.back_days]
    print(a)
    p = scaler.inverse_transform(p)
    m = np.concatenate((a,p),axis=0)

    train['Predictions'] = m
    # print(m)
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Prise USD ($)', fontsize=18)
    plt.plot(train[['Close','Predictions']])
    plt.plot(valid['Close'])
    plt.legend(['Train','Val','Predictions'], loc='lower right')
    plt.savefig('2.jpg')
    return



def test(args,model,test_loader,new_data,device):
    # state_dict = torch.load(args.checkpoint)
    # model_new = LSTM(args).to(device)
    # model_new.load_state_dict(state_dict,strict = True)
    # # model_new = torch.load(args.checkpoint).to(device)
    # model_new.eval()
    model_new=model.eval()

    pre = []
    with torch.no_grad():
        for j,(input, target) in enumerate(test_loader):
            input, target = torch.tensor(input,dtype=torch.float32).to(device), torch.tensor(target,dtype=torch.float32).to(device)
            # print(input)
            predict = model_new.forward(input)
            pre.append([predict.item()])
    train = new_data[:int(len(new_data)*0.8)]
    valid = new_data[int(len(new_data)*0.8):]
    print("========================")
    pre = scaler.inverse_transform(pre)
    # print(pre)
    valid['Predictions'] = pre
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Prise USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close','Predictions']])
    plt.legend(['Train','Val','Predictions'], loc='lower right')
    plt.savefig('3.jpg')
    return



if __name__ == '__main__':
    args = parse_args()
    device = "cuda:"+ str(args.gpu) if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    df = pd.read_csv(args.input_data)
    new_data = df.loc[:,['Date','Close']]
    new_data.index = new_data.Date
    new_data = new_data.sort_index(ascending=True)
    new_data.drop('Date', axis=1, inplace=True)
    # new_data = list(new_data['Close'].values)
    input = np.array(new_data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_rate = 0.8


    #归一化
    input = scaler.fit_transform(input)
    # test_data = scaler.transform(test_data)
    # train_data = new_data
    train_data = input[:int(len(input)*train_rate)]
    test_data = input[int(len(input)*train_rate)-args.back_days:]

    # train_data = GetDataset(args,train_data.values)
    # test_data = GetDataset(args,test_data.values)
    train_data = GetDataset(args,train_data)
    test_data = GetDataset(args,test_data)
    train_loader = DataLoader(train_data, args.batch_size, drop_last = True)
    test_loader = DataLoader(test_data, args.batch_size, drop_last = True)

    model = LSTM(args).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.CrossEntropyLoss()

    train(args, train_loader,criterion,optimizer,new_data,device)
    test(args,model,test_loader,new_data,device)