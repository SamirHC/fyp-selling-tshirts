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
    URL = "https://colorhunt.co"

    @staticmethod
    def download_html(max_scroll=-1):
        service = Service(CHROMEDRIVER_PATH)
        driver = webdriver.Chrome(service=service)

        driver.get(ColorHuntPageScraper.URL)

        time.sleep(2)

        # Accept cookies
        try:
            close_button = driver.find_element(By.XPATH, "//button[@class=\"fc-button fc-cta-consent fc-primary-button\"]")
            close_button.click()
        except Exception as e:
            print("Close button not found")

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
        driver.quit()

        file_path = os.path.join(ColorHuntPageScraper.BASE_HTML_DIR, f"Color Hunt {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.html")
        with open(file_path, "w+", encoding="utf-8") as file:
            file.write(html_content)

        return file_path

    @staticmethod
    def scrape_html_to_dataframe(html_path: str) -> pd.DataFrame:
        with open(html_path, "r") as f:
            tree = etree.HTML(f.read())
        
        palette_divs = tree.xpath("//div[@class=\"item\"]")
        data = []

        service = Service(CHROMEDRIVER_PATH)
        driver = webdriver.Chrome(service=service)

        for i, div in enumerate(palette_divs):
            try:
                palette_id = div.get("data-code")
                palette_url = "/".join((ColorHuntPageScraper.URL, "palette", palette_id))

                driver.get(palette_url)
                time.sleep(0.5)

                tree = etree.HTML(driver.page_source).xpath("//div[@class=\"single hide\"]")[0]
                row = {
                    "palette_id": palette_id,
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
                    "url": palette_url,
                }
                data.append(row)
            except Exception as e:
                print(f"ColorHuntPageScraper: {i}, {e}")

        driver.quit()

        return pd.DataFrame(data)


if __name__ == "__main__":
    file_path = ColorHuntPageScraper.download_html(1)
    print(ColorHuntPageScraper.scrape_html_to_dataframe(file_path))
