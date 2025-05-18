from datetime import datetime
import os
import time

from lxml import etree
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

from src.data_collection.page_scraper import PageScraper
from src.common import config


class ColorHuntPageScraper(PageScraper):
    BASE_HTML_DIR = os.path.join("data", "html", "color_hunt_palettes")
    BASE_SAVE_DIR = os.path.join("data", "dataframes", "color_hunt_palette_data")
    URL = "https://colorhunt.co"

    @staticmethod
    def download_html(max_scroll=-1):
        service = Service(config.CHROMEDRIVER_PATH)
        driver = webdriver.Chrome(service=service)

        driver.get(ColorHuntPageScraper.URL)

        # Accept cookies
        try:
            time.sleep(2)
            close_button = driver.find_element(By.XPATH, "//button[@class=\"fc-button fc-cta-consent fc-primary-button\"]")
            close_button.click()
        except Exception as e:
            print("Close button not found")

        last_height = driver.execute_script("return document.body.scrollHeight")

        while max_scroll:
            driver.find_element(By.TAG_NAME,'body').send_keys(Keys.END)

            time.sleep(0.5)
            new_height = driver.execute_script("return document.body.scrollHeight")

            if new_height == last_height:
                break
            else:
                last_height = new_height
                max_scroll -= 1

        html_content = driver.page_source
        driver.quit()

        file_path = os.path.join(ColorHuntPageScraper.BASE_HTML_DIR, f"Color Hunt {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.html")
        with open(file_path, "w+", encoding="utf-8") as file:
            file.write(html_content)

        return file_path

    @staticmethod
    def scrape_html_to_dataframe(html_path: str) -> pd.DataFrame:
        with open(html_path, "r") as f:
            tree = etree.HTML(f.read())
        
        # Reverse for chronological order
        palette_divs = tree.xpath("//div[@class=\"item\"]")[::-1]
        data = []

        service = Service(config.CHROMEDRIVER_PATH)
        options = Options()
        options.add_argument("--headless")
        driver = webdriver.Chrome(options=options, service=service)

        for n, div in enumerate(palette_divs):            
            palette_id = div.get("data-code")
            palette_url = "/".join((ColorHuntPageScraper.URL, "palette", palette_id))
            while True:
                try:
                    driver.get(palette_url)
                    print(n, palette_id)
                    tree = etree.HTML(driver.page_source).xpath("//div[@class=\"single hide\"]")[0]
                    row = {
                        "palette_id": palette_id,
                        "colors": [
                            tree.xpath(f".//div[@class=\"place c{i}\"]/span")[0].text
                            for i in range(4)
                        ],
                        "likes": int(tree.xpath(".//div[@class=\"button like\"]/span")[0].text.replace(",", "")),
                        "color_tags": [
                            tag.get("tag")
                            for tag in tree.xpath(".//div[@class=\"tags\"]/a")
                            if tag.get("type") == "color"
                        ],
                        "other_tags": [
                            tag.get("tag")
                            for tag in tree.xpath(".//div[@class=\"tags\"]/a")
                            if tag.get("type") != "color"
                        ],
                        "date": tree.xpath(".//span[@class=\"date\"]")[0].text,
                        "url": palette_url,
                    }
                    data.append(row)
                    break
                except Exception as e:
                    print(f"ColorHuntPageScraper: {n, palette_id, e}")

        driver.quit()

        return pd.DataFrame(data)


if __name__ == "__main__":
    file_path = ColorHuntPageScraper.download_html(10)
    print(ColorHuntPageScraper.scrape_html_to_dataframe(file_path))
