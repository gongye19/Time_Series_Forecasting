from bs4 import BeautifulSoup#解析requests请求到的HTML页面
import requests#请求目标网页
import csv
import pandas as pd
from selenium.webdriver import Chrome
from selenium.webdriver.common.keys import Keys   # 键盘的按钮指令
from selenium.webdriver.common.by import By
import time
import json
import re

'''登录'''
url = 'https://www.coinmarketcap.com/'
header = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/\
          70.0.3538.25 Safari/537.36 Core/1.70.3861.400 QQBrowser/10.7.4313.400"}
web = Chrome()
web.get(url)
web.maximize_window()
time.sleep(3)
web.find_element(By.XPATH,'//*[@id="__next"]/div/div[1]/div[1]/div/div[1]/div/div[1]/div/div[3]/button[1]').click()
time.sleep(3)

web.find_element(By.XPATH,'/html/body/div[3]/div/div/div/div/div[2]/div[1]/input').send_keys('gy723511500@gmail.com')
web.find_element(By.XPATH,'/html/body/div[3]/div/div/div/div/div[2]/div[2]/div[2]/input').send_keys('Zh19970119',Keys.ENTER)

time.sleep(3)
web.find_element(By.XPATH,'/html/body/div[3]/div/div/div/div/div[2]/div/form/div[3]/p').click()
input('是否登录完成？')
web.find_element(By.XPATH,'//*[@id="__next"]/div/div[1]/div[1]/div/div[1]/div/div[2]/div[2]/div/div[5]').click()
time.sleep(3)
web.find_element(By.XPATH,'//*[@id="__next"]/div/div[2]/main/div[2]/div[1]/div[1]/div[2]/div[2]/div/div[1]/span').click()
time.sleep(5)


def clean(text):
    # text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", text)  # 去除正文中的@和回复/转发中的用户名
    # text = re.sub(r"\[\S+\]", "", text)      # 去除表情符号
    # text = re.sub(r"#\S+#", "", text)      # 保留话题内容
    p = re.compile(u'['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF' u'\u2600-\u2B55 \U00010000-\U0010ffff]+')
    text = re.sub(p,' ',text)
    # text = re.sub(r'#[A-Za-z]+', '', text)
    # text = re.sub(r'@[A-Za-z]+', '', text)
    text = text.replace('Web link','')
    text = text.replace('Weblink','')
    text = re.sub(r"\s+", " ", text) # 合并正文中过多的空格
    # text = text.replace('Read on:','')
    # text = text.replace('Learn more here:','')
    # text = text.replace('Details','')
    return text.strip()


'''获取username, content, time'''
dvi_list = []
uname = []
con = []
tistamp = []
while True:
    if len(dvi_list)>=50:
        break
    else:
        dvi_list_temp = web.find_elements(By.XPATH,'//*[@id="__next"]/div/div[2]/main/div[2]/div[1]/div[1]/div[2]/div[4]/div/div')
        for i in dvi_list_temp:
            if str(i) not in dvi_list:
                username = i.find_element(By.XPATH,'./div/div/div[2]/div/div/a[2]/span').text
                print(username)
                uname.append(username)
                content = i.find_element(By.XPATH,'./div/div/div[3]/div/div/div').text
                print(content)
                con.append(clean(content))
                timestamp = i.find_element(By.XPATH,'./div/div/div[2]/div/div/span').text
                print(timestamp)
                timestamp = timestamp.replace('·','')
                tistamp.append(timestamp.strip())
                dvi_list.append(str(i))
    web.execute_script("window.scrollBy(0,{})".format(500))
    time.sleep(3)

'''数据处理与保存'''
print(uname)
print(con)
print(tistamp)
print(len(dvi_list))




raw_data = pd.DataFrame(columns = ['username','text','timestamp'])
raw_data['username'] = uname
raw_data['text'] = con
raw_data['timestamp'] = tistamp
print(raw_data.head())
raw_data.to_csv('coinmarket.csv',index = 0)
time.sleep(180)