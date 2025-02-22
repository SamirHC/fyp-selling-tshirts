from datetime import datetime
import os
import time

from dotenv import dotenv_values
from lxml import etree
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service

from src.data_collection.page_scraper import PageScraper


config = dotenv_values(".env")
CHROMEDRIVER_PATH = config["CHROMEDRIVER_PATH"]


class ColorHuntPageScraper(PageScraper):
    BASE_HTML_DIR = os.path.join("data", "html", "color_hunt_palettes")
    BASE_SAVE_DIR = os.path.join("data", "dataframes", "color_hunt_palette_data")

    @staticmethod
    def download_html(max_scroll=-1):
        service = Service(CHROMEDRIVER_PATH)
        driver = webdriver.Chrome(service=service)

        URL = "https://colorhunt.co"
        driver.get(URL)

        time.sleep(2)

        last_height = driver.execute_script("return document.body.scrollHeight")

        while max_scroll:
            driver.find_element(By.TAG_NAME,'body').send_keys(Keys.END)

            time.sleep(2)
            new_height = driver.execute_script("return document.body.scrollHeight")

            if new_height == last_height:
                break
            else:
                last_height = new_height
                max_scroll -= 1

        html_content = driver.page_source

        file_path = os.path.join(ColorHuntPageScraper.BASE_HTML_DIR, f"Color Hunt {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.html")
        with open(file_path, "w+", encoding="utf-8") as file:
            file.write(html_content)

        driver.quit()

    @staticmethod
    def scrape_html_to_dataframe(html_path: str) -> pd.DataFrame:
        with open(html_path, "r") as f:
            tree = etree.HTML(f.read())
            divs = tree.xpath("//div[@class=\"item\"]")

        data = []

        service = Service(CHROMEDRIVER_PATH)
        driver = webdriver.Chrome(service=service)

        for i, div in enumerate(divs):
            try:
                url = div.xpath(".//a")[0].get("href")

                driver.get(url)
                time.sleep(0.5)

                tree = etree.HTML(driver.page_source).xpath("//div[@class=\"single hide\"]")[0]
                row = {
                    "palette_id": tree.xpath(".//div[@class=\"item\"]")[0].get("data-code"),
                    "colors": [
                        tree.xpath(f".//div[@class=\"place c{i}\"]/span")[0].text
                        for i in range(4)
                    ],
                    "likes": tree.xpath(".//div[@class=\"button like\"]/span")[0].text,
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
                    "url": url,
                }
                data.append(row)
            except Exception as e:
                print(i, e)
        
        driver.quit()

        return pd.DataFrame(data)


if __name__ == "__main__":
    ColorHuntPageScraper.download_html(3)
    #print(ColorHuntPageScraper.scrape_directory_to_dataframe())
