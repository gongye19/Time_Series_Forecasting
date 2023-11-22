import ccxt
import pandas as pd
import time
import numpy as np
import os


'''实例化交易所'''
exchange = ccxt.binance({
    'apiKey':'U449GUNIVRevR8rOEJA1ecyxiPGrmEOX9fllyFkZxTWx56xeZPw271zXNBvaBnNs',
    'secret':'ADySAUucXVTwNv32xx2U2yVILQGXgB6SOjahUtyYLCTd57NIxYWlNCwALB32LXxd',
    'timeout': 30000,
    'enableRateLimit': True,
})


'''获取k线数据'''
def get_kline(symbol,timeframe, since):
    kline_data = exchange.fetch_ohlcv(symbol,timeframe,since,limit =1000)
    df_kline = pd.DataFrame(kline_data,columns=['time','open','high','low','close','volume'])
    # df_kline['time'] = pd.to_datetime(df_kline['time'],unit = 'ms')
    df_kline['timestamp'] = df_kline['time'].apply(lambda x:time.localtime(int(x/1000)))
    df_kline['timestamp'] = df_kline['timestamp'].apply(lambda x:time.strftime('%Y-%m-%d %H:%M:%S',x))
    df_kline['symbol'] = symbol
    # df_kline = df_kline.set_index('time')
    # df_kline = df_kline.sort_index(ascending = False)
    return df_kline

'''获取ticker数据'''
def get_ticker(symbol, num):
    ticker_data = exchange.fetch_ticker(symbol)
    df_ticker = pd.DataFrame(ticker_data,index = [num])
    df_ticker['timestamp'] = df_ticker['timestamp'].apply(lambda x:time.localtime(int(x/1000)))
    df_ticker['timestamp'] = df_ticker['timestamp'].apply(lambda x:time.strftime('%Y-%m-%d %H:%M:%S',x))
    return df_ticker

'''获取tickers数据'''
def get_tickers(symbols):
    tickers_df = pd.DataFrame()
    for index,value in enumerate(symbols):
        ticker_df = get_ticker(value,index)
        tickers_df = tickers_df.append(ticker_df)
    return tickers_df

'''获取账户余额'''
def get_account(symbols):
    balance = exchange.fetch_balance()
    balance_df = pd.DataFrame(columns = ['timestamp','symbol','free','used','total'])
    timestamp = []
    sym = []
    free = []
    used = []
    total = []
    for i in symbols:
        sym.append(i)
        timestamp.append(balance['timestamp'])
        free.append(balance[i]['free'])
        used.append(balance[i]['used'])
        total.append(balance[i]['total'])
    balance_df['timestamp'] = timestamp
    balance_df['symbol'] = sym
    balance_df['free'] = free
    balance_df['used'] = used
    balance_df['total'] = total
    balance_df['timestamp'] = balance_df['timestamp'].apply(lambda x:time.localtime(int(x/1000)))
    balance_df['timestamp'] = balance_df['timestamp'].apply(lambda x:time.strftime('%Y-%m-%d %H:%M:%S',x))
    balance_df = balance_df.set_index('timestamp')
    return balance_df

'''获取订单信息'''
def get_orders(symbol):
    open_data = exchange.fetch_orders(symbol)
    if len(open_data) == 0:
        return '无订单!'
    else:
        open_order_df = pd.DataFrame(columns = ['id','timestamp','symbol','type','side','price','amount','status'])
        id = []
        timestamp = []
        sym =[]
        type = []
        side = []
        price = []
        amount = []
        status = []
        for i in range(len(open_data)):
            open_order = open_data[i]
            id.append(open_order['id'])
            timestamp.append(open_order['timestamp'])
            sym.append(open_order['symbol'])
            type.append(open_order['type'])
            side.append(open_order['side'])
            price.append(open_order['price'])
            amount.append(open_order['amount'])
            status.append(open_order['status'])
        open_order_df['id']= id
        open_order_df['timestamp']=timestamp
        open_order_df['symbol']=sym
        open_order_df['type']=type
        open_order_df['side']=side
        open_order_df['price']=price
        open_order_df['amount']=amount
        open_order_df['status']=status
        open_order_df['timestamp'] = open_order_df['timestamp'].apply(lambda x:time.localtime(int(x/1000)))
        open_order_df['timestamp'] = open_order_df['timestamp'].apply(lambda x:time.strftime('%Y-%m-%d %H:%M:%S',x))
        open_order_df = open_order_df.set_index('timestamp')
        open_order_df = open_order_df.sort_index(ascending = False)
        return open_order_df



def main():
    currancy = ['USDT','BTC','ETH','BNB','DOGE']
    symbol = 'BTC/USDT'
    symbols = ['BTC/USDT','ETH/USDT','BNB/USDT','DOGE/USDT']
    timeframe = '5m'



    
    tickers = get_tickers(symbols)
    orders = get_orders(symbol)
    account = get_account(currancy)
    # 1676725800000
    # os.getcwd()
    # since = 1673725800000
    # for i in range(10000):
    #     kline = get_kline(symbol,timeframe,since)
    #     since = since + 300000000
    #     kline.to_csv('E:/trade/BTC.csv',mode = 'a',header = 0,index = 0)
    #     print('============================================')
    #     time.sleep(1)


    
    # print(kline)
    print(tickers)
    print(orders)
    print(account)



    # 限价单
    # buy = exchange.create_limit_buy_order(symbol,0.002,8000)
    # sell = exchange.create_limit_sell_order (symbol, amount, price)

    # 市价单
    # sell = exchange.create_market_sell_order (symbol, amount)
    # buy = exchange.create_market_buy_order (symbol, amount)

    # 撤销订单
    # exchange.cancelOrder(id,symbol)
    
    
if __name__ == '__main__':
    # while True:
    main()
        # time.sleep(3)