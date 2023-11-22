import requests
import pandas as pd

pd.set_option('expand_frame_repr', False)

def get_ticker(idx,symbol):

    url = 'https://api.binance.com/api/v3/ticker?symbol={}'.format(symbol)
    try:
        res = requests.get(url,timeout=15)
    except Exception as e:
        print("错误!", e)
        return None
    res = res.json()
    raw_data = pd.DataFrame(res,index=[idx])
    raw_data['openTime'] = pd.to_datetime(raw_data['openTime'], unit = 'ms')
    raw_data['closeTime'] = pd.to_datetime(raw_data['closeTime'], unit = 'ms')
    return raw_data


def get_tickers(symbols):
    tickers_df = pd.DataFrame()
    for index,value in enumerate(symbols):
        ticker_df = get_ticker(index,value)
        tickers_df = tickers_df.append(ticker_df)
    return tickers_df



def get_kline(symbol,interval='5m',limit=100):
    url = 'https://api.binance.com/api/v3/klines?symbol={}&interval={}&limit={}'.format(symbol,interval,limit)
    try:
        res = requests.get(url,timeout=15)
    except Exception as e:
        print("错误!", e)
        return None
    res = res.json()
    raw_data = pd.DataFrame(res)
    data = raw_data.copy()
    data.columns = ['Kline open time','Open price','High price','Low price','Close price','Volume','Kline Close time','Quote asset volume',\
    'Number of trades','Taker buy base asset volume','Taker buy quote asset volume','Symbol']
    data['Kline open time'] = pd.to_datetime(data['Kline open time'],unit = 'ms')
    data['Symbol'] = symbol
    data = data.set_index('Kline open time')
    return data
    


def main():
    symbols = ['BTCUSDT','BNBUSDT','ETHUSDT','DOGEUSDT']
    tickers = get_tickers(symbols)
    # print(tickers)
    line = get_kline('BTCUSDT')
    print(line)


if __name__ == '__main__':
    main()
