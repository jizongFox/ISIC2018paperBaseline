#coding=utf8
from selenium import webdriver
import time

def slow_input(ele,str):
    '''减慢账号密码的输入速度'''
    for i in str:
        ele.send_keys(i)
        time.sleep(0.5)

driver = webdriver.Chrome()
page = 'https://challenge.kitware.com/#phase/5b1c193356357d41064da2ec/submit'

driver.get(page)
time.sleep(3)
id = driver.find_element_by_id('g-login')
id.clear()
print('input login name')
id.send_keys('jizong')
print('input password')
pw = driver.find_element_by_id('g-password')
pw.clear()
pw.send_keys('911005')
print('login')
login = driver.find_element_by_id('g-login-button')
login.click()
time.sleep(3)


driver.close()