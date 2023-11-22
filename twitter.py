from bs4 import BeautifulSoup#解析requests请求到的HTML页面
import requests#请求目标网页
import csv

from selenium.webdriver import Chrome
from selenium.webdriver.common.keys import Keys   # 键盘的按钮指令
from selenium.webdriver.common.by import By
import time
import json


url = 'https://www.twitter.com/'
header = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/\
          70.0.3538.25 Safari/537.36 Core/1.70.3861.400 QQBrowser/10.7.4313.400"}
web = Chrome()
web.get(url)
web.maximize_window()
with open('twitter_cookies.txt', 'r', encoding='utf8') as f:
        listCookies = json.loads(f.read())
 
# 往browser里添加cookies
for cookie in listCookies:
    if 'expiry' in cookie:
            del cookie['expiry']
    
    web.add_cookie(cookie)
web.refresh() 

time.sleep(3)
web.find_element(By.XPATH,'//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[2]\
                 /div/div[2]/div/div/div/div[1]/div/div/div/form/div[1]/div/div/div/label/div[2]/div/input').send_keys('cryptocurrency',Keys.ENTER)

time.sleep(3)
web.find_element(By.XPATH,'//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[1]/div/div[1]/div[1]/div[2]/nav/div/div[2]/div/div[2]/a/div/div/span').click()
time.sleep(180)