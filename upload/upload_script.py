#coding=utf8
from selenium import webdriver
import time,requests,pickle

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Referer': 'https://consumeprod.alipay.com/record/advanced.htm',
    'Host': 'consumeprod.alipay.com',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Connection': 'keep-alive'
}


class automatic_upload(object):
    def __init__(self) -> None:
        super().__init__()
        self.page_to_start = 'https://challenge.kitware.com/#phase/5b1c193356357d41064da2ec/submit'
        self.id = 'jizong'
        self.pw = '911005'
        self.session = requests.Session()
        self.header = HEADERS
        self.session.headers = self.header
        self.implicitly_wait_time =30
        self.cookies_dict={}


    def slow_input(self,ele,str):
        '''减慢账号密码的输入速度'''
        for i in str:
            ele.send_keys(i)
            time.sleep(0.2)

    def get_cookies(self):
        cookies = self.driver.get_cookies()
        # cookies_dict = {}
        # for cookie in cookies:
        #     if 'name' in cookie and 'value' in cookie:
        #         cookies_dict[cookie['name']] = cookie['value']
        #
        # self.cookies_dict = cookies_dict
        # pickle.dump(self.driver.get_cookies(), open("cookies.pkl", "wb"))
        self.cookies_dict=cookies
    def set_cookies(self):
        '''将获取到的cookies加入session'''
        # cookies = pickle.load(open("cookies.pkl", "rb"))
        # self.driver.add_cookie(self.cookies_dict)
        for cookie in self.cookies_dict:
            self.driver.add_cookie(cookie)

    def fill_login(self):
        self.driver = webdriver.Chrome()
        self.driver.implicitly_wait(self.implicitly_wait_time)
        self.driver.get(self.page_to_start)
        time.sleep(3)
        id = self.driver.find_element_by_id('g-login')
        print('input login name')
        self.slow_input(id,self.id)
        print('input password')
        pw = self.driver.find_element_by_id('g-password')
        pw.clear()
        self.slow_input(pw,self.pw)
        print('login')
        login = self.driver.find_element_by_id('g-login-button')
        login.click()
        # time.sleep(3)
        self.get_cookies()
        self.driver.close()

    def upload(self):
        self.driver = webdriver.Chrome()
        # self.set_cookies()
        self.driver.get(self.page_to_start)
        time.sleep(1)
        self.set_cookies()
        self.driver.get(self.page_to_start)
        time.sleep(10)





if __name__=="__main__":
    a = automatic_upload()
    a.fill_login()
    time.sleep(2)
    a.upload()
