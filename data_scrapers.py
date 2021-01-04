import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

options = Options()
options.headless = True
options.add_argument('--window-size=1920,1200')

driver = webdriver.Chrome(options=options, executable_path='/Users/Admin/UserDrivers/chromedriver')

def get_yf_gainers(quantity, price):
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
