# coding=utf8
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time, requests, pickle, os, pandas as pd
from io import StringIO

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Referer': 'https://consumeprod.alipay.com/record/advanced.htm',
    'Host': 'consumeprod.alipay.com',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Connection': 'keep-alive'
}


class automatic_upload(object):
    def __init__(self,filename) -> None:
        super().__init__()
        # self.page_to_start = 'https://challenge.kitware.com/#phase/5b1c193356357d41064da2ec'  # final test
        self.page_to_start = 'https://challenge.kitware.com/#phase/5b1c16ca56357d41064da2e7'
        self.id = 'jizong'
        self.pw = '911005'
        self.session = requests.Session()
        self.header = HEADERS
        self.session.headers = self.header
        self.implicitly_wait_time = 5
        self.cookies_dict = {}
        self.filename= filename

    def slow_input(self, ele, str):
        '''减慢账号密码的输入速度'''
        ele.send_keys(Keys.DELETE)
        for i in str:
            ele.send_keys(i)
            time.sleep(0.1)

    def get_cookies(self):
        cookies = self.driver.get_cookies()
        self.cookies_dict = cookies

    def set_cookies(self):
        '''将获取到的cookies加入session'''
        # cookies = pickle.load(open("cookies.pkl", "rb"))
        # self.driver.add_cookie(self.cookies_dict)
        for cookie in self.cookies_dict:
            self.driver.add_cookie(cookie)

    def fill_login(self):
        options = webdriver.ChromeOptions()
        options.add_argument('headless')

        self.driver = webdriver.Chrome(chrome_options=options)
        self.driver.implicitly_wait(self.implicitly_wait_time)
        self.driver.get(self.page_to_start)
        time.sleep(1)
        button = self.driver.find_element_by_id('c-join-phase')
        button.click()
        time.sleep(1)
        id = self.driver.find_element_by_id('g-login')
        print('input login name')
        time.sleep(1)
        self.slow_input(id, self.id)
        time.sleep(1)
        print('input password')
        pw = self.driver.find_element_by_id('g-password')
        self.slow_input(pw, self.pw)
        print('login')
        login = self.driver.find_element_by_id('g-login-button')
        login.click()
        print()
        time.sleep(2)
        button = self.driver.find_element_by_id('c-submit-phase-dataset')
        button.click()

        self.upload(fill=False)

    def upload(self, fill):
        time.sleep(2)
        if fill:
            print('descriptor')
            try:
                description = self.driver.find_element_by_xpath(
                    '//*[@id="g-app-body-container"]/div[2]/div[2]/div[1]/div/select')
            except Exception as e:
                description = self.driver.find_element_by_xpath(
                    '//*[@id="g-app-body-container"]/div[2]/div[2]/div[1]/div/input')

            self.slow_input(description, '111')
            # description.send_keys('1111')
            teamname = self.driver.find_element_by_class_name('c-submission-organization-input')
            self.slow_input(teamname, 'dahuli')
            # teamname.send_keys('dahuli')
            url = self.driver.find_element_by_class_name('c-submission-organization-url-input')
            self.slow_input(url, 'dahuli.com')
            # url.send_keys('www.dahuli.com')
            try:
                arxiv = self.driver.find_element_by_class_name('c-submission-documentation-url-input')
                # arxiv.send_keys('www.arxiv.dahuli.com')
                self.slow_input(arxiv, 'www.arxiv.dahuli.com')
            except:
                pass
            externaldata = self.driver.find_element_by_xpath(
                '//*[@id="g-app-body-container"]/div[2]/div[2]/label[2]/input')
            time.sleep(0.5)
            externaldata.click()

        ## agreement
        try:
            agree = self.driver.find_element_by_xpath(
                '//*[@id="g-app-body-container"]/div[2]/div[2]/div[5]/label/input')
        except:
            agree = self.driver.find_element_by_xpath(
                '//*[@id="g-app-body-container"]/div[2]/div[2]/div[4]/label/input')
        time.sleep(0.2)
        agree.click()

        file = self.driver.find_element_by_id('g-files')
        file.send_keys(os.path.join(os.getcwd(), 'val.zip'))
        time.sleep(1)
        submit = self.driver.find_element_by_xpath('//*[@id="g-upload-form"]/div[8]/button')
        submit.click()
        # time.sleep(100)

    def get_excel(self):
        while True:
            if self.driver.page_source.find('Overall score:') > 0:
                print('We have the overall score')
                break
            else:
                time.sleep(2)

        excel = self.driver.find_element_by_xpath('//*[@id="g-app-body-container"]/div[2]/div[2]/table/tbody').text
        excel = pd.read_csv(StringIO(excel), )
        excel['image_id']=excel.iloc[:,0].apply(lambda x:x.split(' ')[0])
        excel['image_score'] = excel.iloc[:,0].apply(lambda x:x.split(' ')[1])
        excel.to_csv(self.filename.replace('zip','csv'),encoding='utf8')
        self.driver.close()


if __name__ == "__main__":
    a = automatic_upload('val.zip')
    a.fill_login()
    a.get_excel()
