import os

from lxml import etree
import pandas as pd

from src.data_collection.page_scraper import PageScraper


class EtsyPageScraper(PageScraper):
    BASE_HTML_DIR = os.path.join("data", "html", "etsy_listings")
    BASE_SAVE_DIR = os.path.join("data", "dataframes", "etsy_listing_data")

    @staticmethod
    def scrape_html_to_dataframe(html_path: str) -> pd.DataFrame:
        """
        Extracts the item sales data from an Etsy HTML page.

        :param html_path: Etsy HTML page file path

        :return df: dataframe of extracted item sales data
        """

        with open(html_path, "r") as f:
            tree = etree.HTML(f.read())
            ul = tree.xpath("//ul[contains(@class, \"wt-grid wt-grid--block wt-pl-xs-0 tab-reorder-container\")]")[0]
            lis = ul.findall("li")

        data = []

        for li in lis:
            try:
                item_id = li.xpath(".//div[contains(@class, \"v2-listing-card\")]")[0].get("data-listing-id")

                listing_card_info = li.xpath(".//div[contains(@class, \"v2-listing-card__info\")]")[0]
                seller_rating_info = listing_card_info.xpath(".//span[contains(@class, \"larger_review_stars\")]")[0]
                price_elem = listing_card_info.xpath(".//p[contains(@class, \"lc-price\")]")[0]

                data.append({
                    "item_id": item_id,
                    "title": listing_card_info.xpath(".//h3")[0].text.strip().replace("\n", ""),
                    "item_url": f"https://www.etsy.com/uk/listing/{item_id}",
                    "img_url": li.xpath(".//img")[0].get("data-preload-lp-src"),
                    "seller_star_rating": float(seller_rating_info.xpath(".//span[contains(@class, \"wt-text-title-small\")]")[0].text),
                    "seller_review_count": seller_rating_info.xpath(".//p[contains(@class, \"wt-text-body-smaller\")]")[0].text.strip(),
                    "currency_symbol": price_elem.xpath(".//span[contains(@class, \"currency-symbol\")]")[0].text,
                    "price": price_elem.xpath(".//span[contains(@class, \"currency-value\")]")[0].text,
                })
            except Exception as e:
                print(e)

        return pd.DataFrame(data)


if __name__ == "__main__":
    df = EtsyPageScraper.scrape_directory_to_dataframe()
    print(df)
