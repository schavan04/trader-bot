import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException

def get_yf_gainers(price):
    options = Options()
    options.headless = True
    options.add_argument("window-size=1920,1080")
    driver = webdriver.Chrome(options=options, executable_path='/Users/Admin/UserDrivers/chromedriver')
    try:
        driver.get('https://finance.yahoo.com/gainers')

        change_button = driver.find_element_by_xpath('//*[@id="scr-res-table"]/div[1]/table/thead/tr/th[4]')
        change_button.click()
        time.sleep(2)

        unswitched = True
        while unswitched:
            change_button.click()
            time.sleep(2)

            caret = driver.find_element_by_css_selector('#scr-res-table > div.Ovx\(a\).Ovx\(h\)--print.Ovy\(h\) > table > thead > tr > th.Ta\(end\).Pstart\(20px\).Bgc\(\$lv3BgColor\).Fz\(xs\).Va\(m\).Py\(5px\)\!.Cur\(p\).Bgc\(\$hoverBgColor\)\:h.C\(\$primaryColor\).Fw\(500\) > svg')
            if caret.get_attribute('data-icon') == "caret-down":
                unswitched = False

        gainers = driver.find_elements_by_css_selector('td > a.Fw\(600\)')
        prices = driver.find_elements_by_css_selector('td:nth-child(3) > span.Trsdu\(0\.3s\)')

        symbols = []
        if prices:
            for i in range(0, len(gainers)):
                if float(prices[i].text.replace(',', '')) <= price:
                    symbols.append(gainers[i].text)
        return symbols

    finally:
        driver.quit()

def check_rating(symbols, quantity):

    if quantity > len(symbols):
        quantity = len(symbols)

    options = Options()
    options.headless = True
    options.add_argument("window-size=1920,1080")
    driver = webdriver.Chrome(options=options, executable_path='/Users/Admin/UserDrivers/chromedriver')
    try:
        approved = []
        counter = 0
        valid = 0

        while valid < quantity:
            symbol = symbols[counter]
            driver.get(f'https://finance.yahoo.com/quote/{symbol}?p={symbol}&.tsrc=fin-srch')
            time.sleep(1)
            driver.execute_script("window.scrollTo(0, 1080)")
            time.sleep(2)
            try:
                rating = driver.find_element_by_xpath('//*[@id="Col2-8-QuoteModule-Proxy"]/div/section/div/div/div[1]')
                time.sleep(2)
                if float(rating.text) < 2.8:
                    approved.append(symbol)
                    valid += 1
                    print(symbol + " approved for buy")
                else:
                    print(symbol + " not recommended")
            except NoSuchElementException:
                print(symbol + " does not have a recommendation rating")

            counter +=1

        return approved

    finally:
        driver.quit()
