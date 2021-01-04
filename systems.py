import abc

import auth
import data_scrapers
import data_filters

av = auth.av
paper = auth.paper

class ShellSystem:
    def __init__(self, api):
        self.api = api

    def get_data(self):
        scraper = data_scrapers.YFScraper()
        symbols = scraper.get_gainers(quantity=5, price=200)
        return symbols
