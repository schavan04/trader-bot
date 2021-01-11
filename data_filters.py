import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

options = Options()
options.headless = False
options.add_argument("window-size=1920,1080")

def get_yf_gainers(quantity, price):
    driver = webdriver.Chrome(options=options, executable_path='/Users/Admin/UserDrivers/chromedriver')
    try:
        driver.get('https://finance.yahoo.com/gainers')

        change_button = driver.find_element_by_xpath('//*[@id="scr-res-table"]/div[1]/table/thead/tr/th[4]')
        change_button.click()
        time.sleep(2)
        change_button.click()
        time.sleep(2)

        gainers = driver.find_elements_by_css_selector('td > a.Fw\(600\)')
        prices = driver.find_elements_by_css_selector('td:nth-child(3) > span.Trsdu\(0\.3s\)')

        symbols = []
        valid = 0
        counter = 0
        while valid < quantity:
            if float(prices[counter].text.replace(',', '')) <= price:
                symbols.append(gainers[counter].text)
                valid += 1
            counter += 1

        return symbols
    finally:
        driver.quit()

def check_rating(symbols):
    driver = webdriver.Chrome(options=options, executable_path='/Users/Admin/UserDrivers/chromedriver')
    try:
        approved = []
        for symbol in symbols:
            driver.get(f'https://finance.yahoo.com/quote/{symbol}?p={symbol}&.tsrc=fin-srch')
            time.sleep(1)
            driver.execute_script("window.scrollTo(0, 1080)")
            time.sleep(2)
            rating = driver.find_element_by_xpath('//*[@id="Col2-8-QuoteModule-Proxy"]/div/section/div/div/div[1]')
            time.sleep(2)
            if float(rating.text) < 2.4:
                approved.append(symbol)
                print(symbol + " approved for buy")
            else:
                print(symbol + " not recommended")
        return approved
    finally:
        driver.quit()
