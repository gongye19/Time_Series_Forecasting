import pandas as pd
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from model import LSTM
from sklearn.preprocessing import MinMaxScaler
import argparse



# back_days过多会使得预测值偏低，不准确

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, default='./NSE-TATAGLOBAL11.csv')
    parser.add_argument('--save_path', type=str, default='./checkpoint')
    parser.add_argument('--seed', type=int, default=88)
    parser.add_argument('--best_model', type=int, default=500)
    parser.add_argument('--gpu', type=str, default='cuda:0')
    parser.add_argument('--back_days', type=int, default=7)
    parser.add_argument('--predict_days', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--only7', action='store_true', default=False)
    parser.add_argument('--see_train', action='store_true', default=False)
    parser.add_argument('--skip_train', action='store_true', default=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    device=torch.device(args.gpu if torch.cuda.is_available() else 'cpu')

    path = 'E:/trade/BTC.csv'
    col = ['time','open','high','low','close','volume','timestamp','symbol']
    raw_data = pd.read_csv(path,header = None, names = col)
    history_data = raw_data.copy()
    history_data = history_data.set_index('timestamp')
    history_data = history_data.fillna(method='ffill')
    history_data = history_data.drop('time',axis = 1)
    history_data = history_data[['open','high','low','close']]

    '''归一化'''
    scaler = MinMaxScaler(feature_range=(0, 1))
    # 一列一列的归一化，输入数据格式必须是[n,1]形状的数据
    for line in history_data:                           # 这里不能进行统一进行缩放，因为fit_transform返回值是numpy类型
        history_data[line] = scaler.fit_transform(history_data[line].values.reshape(-1,1))
    history_data = history_data.astype(np.float32)

    '''制作训练测试数据'''
    data_feat, data_target = [],[]
    back_days = args.back_days
    predict_days = args.predict_days
    feature_num = 4
    ba_size = args.batch_size
    for index in range(len(history_data) - back_days - predict_days+1):
        # 构建特征集
        data_feat.append(history_data[['open','high','low','close']][index: index + back_days].values)
        # 构建target集
        data_target.append(history_data['close'][index+back_days:index + back_days+predict_days])
    # 将特征集和标签集整理成numpy数组
    data_feat = np.array(data_feat)
    data_target = np.array(data_target)
    # 这里按照8:2的比例划分训练集和测试集
    test_size = int(np.round(0.2*len(history_data)))  # np.round(1)是四舍五入，
    train_size = len(history_data) - (test_size)
    print('训练集:'+str(train_size))     # 输出训练集大小
    print('测试集:'+str(test_size))  # 输出测试集大小
    trainX = torch.Tensor(data_feat[:train_size].reshape(-1,back_days,feature_num)).to(device)
    testX  = torch.Tensor(data_feat[train_size:].reshape(-1,back_days,feature_num)).to(device)
    trainY = torch.Tensor(data_target[:train_size].reshape(-1,predict_days,1)).to(device)
    testY  = torch.Tensor(data_target[train_size:].reshape(-1,predict_days,1)).to(device)


    train = torch.utils.data.TensorDataset(trainX,trainY)
    test = torch.utils.data.TensorDataset(testX,testY)
    train_loader = torch.utils.data.DataLoader(dataset=train,batch_size=ba_size,shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test,batch_size=ba_size,shuffle=False)

    input_dim = feature_num      # 数据的特征数
    hidden_dim = 30    # 隐藏层的神经元个数
    num_layers = 2     # LSTM的层数
    output_dim = predict_days     # 预测值的特征数
    epoch = args.epoch
    train_loss=[]#记录训练loss
    test_loss=[]#记录测试loss
    best_loss = 1
    best_model = args.best_model

    if not args.skip_train:
        '''训练测试'''
        model = LSTM(args,input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        for i in range(epoch):
            loss_nowEpoch=[]
            model.train()
            for idx, (batch_x, batch_y) in enumerate(train_loader):
                out=model(batch_x)#模型输入
                Loss = F.mse_loss(out,batch_y.view(-1,predict_days)).to(device)#loss计算，将batch_y从(64,7,4)变形为(64,28)
                # Loss.to(device)
                optimizer.zero_grad()#当前batch的梯度不会再用到，所以清除梯度
                Loss.backward()#反向传播计算梯度
                optimizer.step()#更新参数
                loss_nowEpoch.append(Loss.item())
                break
            train_loss.append(sum(loss_nowEpoch)/len(loss_nowEpoch))

            loss_nowEpochTest = []
            model.eval()
            for step, (batch_x, batch_y) in enumerate(test_loader):
                test_out = model(batch_x)
                Loss = F.mse_loss(test_out, batch_y.view(-1, predict_days))  # 将batch_y从(64,7,4)变形为(64,28)
                loss_nowEpochTest.append(Loss.item())
                break
            test_loss.append(sum(loss_nowEpochTest)/len(loss_nowEpochTest))

            print(">>> EPOCH:{}, TrainLoss:{:.3f}, TestLoss:{:.3f}".format(i+1, train_loss[-1],test_loss[-1]))

            if i>np.round(epoch*0.9):
                torch.save(model.state_dict(), args.save_path+"/model_{}.pth".format((i+1)))
                best_model = i+1

    "测试集效果图"
    model_new = LSTM(args,input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    model_new.load_state_dict(torch.load(args.save_path+'/model_{}.pth'.format(best_model)),strict = True)
    model_new.eval()
    model_new.to(device)
    y_test_pred = model_new(testX)
    testY = testY.view(-1,args.predict_days)

    pred_value = y_test_pred.cpu().detach().numpy()[:,-1]
    true_value = testY.cpu().detach().numpy()[:,-1]
    pred_value = scaler.inverse_transform(pred_value.reshape(-1, 1))
    true_value = scaler.inverse_transform(true_value.reshape(-1, 1))

    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Prise USD ($)', fontsize=18)
    plt.plot(true_value, label="Data")
    plt.plot(pred_value, label="Preds")
    plt.legend(['True','Predictions'], loc='lower right')
    plt.show()


            