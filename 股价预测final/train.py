import tushare as ts
import pandas as pd
import mplfinance as mpf
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # MODIFIABLE MEVF HYPER-PARAMETERS--------------------------------------------------------------------------------
    # Model loading/saving
    parser.add_argument('--input_data', type=str, default='./NSE-TATAGLOBAL11.csv')
    parser.add_argument('--save_path', type=str, default='./checkpoint')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/model_327.pth')
    parser.add_argument('--seed', type=int, default=88)
    parser.add_argument('--gpu', type=str, default=0)
    parser.add_argument('--back_days', type=int, default=30)
    parser.add_argument('--predict_days', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--only7', action='store_true', default=False)
    parser.add_argument('--see_train', action='store_true', default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    '''获取数据'''
    pro = ts.pro_api('1a11f93fd19add59c3786c056fa579c93304fcad56987ed456bf3ca6')
    df = pro.daily(ts_code='000001.SZ', start_date='20120101', end_date='20230101')
    df = df.set_index('trade_date')
    df.index = pd.to_datetime(df.index)
    df.sort_index(ascending = True, inplace = True)
    data = df[['open','high','low','close']]
    data = data.fillna(method='ffill')
    print(data.tail(args.predict_days))
    '''处理数据'''
    scaler = MinMaxScaler(feature_range=(-1, 1))
    for line in data:                           # 这里不能进行统一进行缩放，因为fit_transform返回值是numpy类型
        data[line] = scaler.fit_transform(data[line].values.reshape(-1,1))
    data = data.astype(np.float32)


    '''制作训练测试数据'''
    data_feat, data_target = [],[]
    seq = args.back_days
    pre = args.predict_days
    feature_num = 4
    ba_size = args.batch_size

    for index in range(len(data) - seq - pre+1):
        # 构建特征集
        data_feat.append(data[['open','high','low','close']][index: index + seq].values)
        # 构建target集
        data_target.append(data['close'][index+seq:index + seq+pre])
    # 将特征集和标签集整理成numpy数组
    data_feat = np.array(data_feat)
    data_target = np.array(data_target)
    # 这里按照8:2的比例划分训练集和测试集
    test_size = int(np.round(0.2*data.shape[0]))  # np.round(1)是四舍五入，
    train_size = data_feat.shape[0] - (test_size)

    print('训练集:'+str(train_size))     # 输出训练集大小
    print('测试集:'+str(test_size))  # 输出测试集大小

    # 这里第一个维度自动确定，我们认为其为batch_size，因为在LSTM类的定义中，设置了batch_first=True
    trainX = torch.from_numpy(data_feat[:train_size].reshape(-1,seq,feature_num)).type(torch.Tensor)
    testX  = torch.from_numpy(data_feat[train_size:].reshape(-1,seq,feature_num)).type(torch.Tensor)
    trainY = torch.from_numpy(data_target[:train_size].reshape(-1,pre,1)).type(torch.Tensor)
    testY  = torch.from_numpy(data_target[train_size:].reshape(-1,pre,1)).type(torch.Tensor)

    train = torch.utils.data.TensorDataset(trainX,trainY)
    test = torch.utils.data.TensorDataset(testX,testY)
    train_loader = torch.utils.data.DataLoader(dataset=train,batch_size=ba_size,shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test,batch_size=ba_size,shuffle=False)
    
    input_dim = feature_num      # 数据的特征数
    hidden_dim = 80    # 隐藏层的神经元个数
    num_layers = 2     # LSTM的层数
    output_dim = args.predict_days     # 预测值的特征数
    epoch = args.epoch
    if args.train:
        '''训练测试'''
        
        model = LSTM(args,input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        lossList=[]#记录训练loss
        lossListTest=[]#记录测试loss
        best_loss = 1
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        for i in range(epoch):
            loss_nowEpoch=[]
            model.train()
            for step, (batch_x, batch_y) in enumerate(train_loader):
                # print(batch_y.shape)
                out=model(batch_x)#模型输入
                Loss = F.mse_loss(out,batch_y.view(-1,args.predict_days))#loss计算，将batch_y从(64,7,4)变形为(64,28)
                optimizer.zero_grad()#当前batch的梯度不会再用到，所以清除梯度
                Loss.backward()#反向传播计算梯度
                optimizer.step()#更新参数
                loss_nowEpoch.append(Loss.item())
                break
            lossList.append(sum(loss_nowEpoch)/len(loss_nowEpoch))

            loss_nowEpochTest = []
            model.eval()
            for step, (batch_x, batch_y) in enumerate(test_loader):
                out = model(batch_x)
                Loss = F.mse_loss(out, batch_y.view(-1, args.predict_days))  # 将batch_y从(64,7,4)变形为(64,28)
                loss_nowEpochTest.append(Loss.item())
                break
            lossListTest.append(sum(loss_nowEpochTest)/len(loss_nowEpochTest))

            print(">>> EPOCH:{}, averTrainLoss:{:.3f}, averTestLoss:{:.3f}".format(i+1, lossList[-1],lossListTest[-1]))

            if len(lossListTest)>2:
                if lossListTest[-1] < best_loss:
                    best_loss = lossListTest[-1]
                    if i>300:
                        torch.save(model.state_dict(), args.save_path+"/model_{}.pth".format((i+1)))

        '''绘制loss'''
        plt.plot(list(range(epoch)),lossList,label='Train')
        plt.plot(list(range(epoch)),lossListTest,label='Test')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig('loss.jpg')



    else:
        if args.see_train:
            model_new = LSTM(args,input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
            model_new.load_state_dict(torch.load(args.checkpoint),strict = True)
            y_train_pred = model_new(trainX)
            trainY = trainY.view(-1,args.predict_days)
            pred_value = y_train_pred.detach().numpy()[:,-1]
            true_value = trainY.detach().numpy()[:,-1]
            pred_value = scaler.inverse_transform(pred_value.reshape(-1, 1))
            true_value = scaler.inverse_transform(true_value.reshape(-1, 1))

            plt.figure(figsize=(16,8))
            plt.title('Model')
            plt.xlabel('Date', fontsize=18)
            plt.ylabel('Close Prise USD ($)', fontsize=18)
            plt.plot(true_value, label="Data")
            plt.plot(pred_value, label="Preds")
            plt.legend(['Train','Predictions'], loc='lower right')

            plt.savefig('train.jpg')

        else:
            if args.only7:
                testX = testX[-1].view(1,args.back_days,feature_num)
                testY = testY[-1]
            '''绘制预测结果'''
            model_new = LSTM(args,input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
            model_new.load_state_dict(torch.load(args.checkpoint),strict = True)
            y_test_pred = model_new(testX)
            testY = testY.view(-1,args.predict_days)
            "测试集效果图"
            if args.only7:
                pred_value = y_test_pred.detach().numpy()
                true_value = testY.detach().numpy()
            else:
                pred_value = y_test_pred.detach().numpy()[:,-1]
                true_value = testY.detach().numpy()[:,-1]

            pred_value = scaler.inverse_transform(pred_value.reshape(-1, 1))
            true_value = scaler.inverse_transform(true_value.reshape(-1, 1))
            if args.only7:
                print(pred_value)
                print(true_value)
            plt.figure(figsize=(16,8))
            plt.title('Model')
            plt.xlabel('Date', fontsize=18)
            plt.ylabel('Close Prise USD ($)', fontsize=18)
            plt.plot(true_value, label="Data")
            plt.plot(pred_value, label="Preds")
            plt.legend(['Train','Predictions'], loc='lower right')
            if args.only7:
                plt.savefig('test7.jpg')

            else:
                plt.savefig('test.jpg')
