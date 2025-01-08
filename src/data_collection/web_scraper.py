from typing import Type
import os
from datetime import datetime

import pandas as pd

from src.common import utils
from src.data_collection.page_scraper import PageScraper
from src.data_collection.ebay_page_scrape import EbayPageScraper
from src.data_collection.etsy_page_scrape import EtsyPageScraper
from src.data_collection.color_hunt_scrape import ColorHuntPageScraper



class WebScraper:
    def __init__(self, page_scraper: Type[PageScraper], save_file=None):
        self.save_file = save_file
        self.page_scraper = page_scraper

    def scrape_data(self):
        new_df = self.page_scraper.scrape_directory_to_dataframe()

        if self.save_file is None:
            self.save_file = f"{self.page_scraper.__name__} {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.pickle"

        path = os.path.join(self.page_scraper.BASE_SAVE_DIR, self.save_file)

        if self.save_file in os.listdir(self.page_scraper.BASE_SAVE_DIR):
            old_df = utils.load_data(path)
            df = pd.concat((old_df, new_df), ignore_index=True)
        else:
            df = new_df

        utils.save_data(df, path)

    @staticmethod
    def ebay_scraper(save_file=None):
        return WebScraper(EbayPageScraper, save_file)

    @staticmethod
    def etsy_scraper(save_file=None):
        return WebScraper(EtsyPageScraper, save_file)

    @staticmethod
    def color_hunt_scraper(save_file=None):
        return WebScraper(ColorHuntPageScraper, save_file)


if __name__ == "__main__":
    WebScraper.ebay_scraper().scrape_data()
    WebScraper.etsy_scraper().scrape_data()
    WebScraper.color_hunt_scraper().scrape_data()
