import os

from lxml import etree
import pandas as pd

from src.data_collection.page_scraper import PageScraper


class EbayPageScraper(PageScraper):
    BASE_HTML_DIR = os.path.join("data", "html", "ebay_seller_hub_research")
    BASE_SAVE_DIR = os.path.join("data", "dataframes", "seller_hub_data")

    @staticmethod
    def scrape_html_to_dataframe(html_path: str) -> pd.DataFrame:
        """
        Extracts the item sales data from an eBay Seller Hub Research HTML page.

        :param html_path: eBay Seller Hub Research HTML page file path

        :return df: dataframe of extracted item sales data
        """

        with open(html_path, "r") as f:
            tree = etree.HTML(f.read())
            rows = tree.xpath("//table/tr[contains(@class, \"research\")]")[1:]

        data = []

        for row in rows:
            try:
                product_info_name = row.xpath(".//div[@class=\"research-table-row__product-info-name\"]/a/span")[0]

                data.append({
                    "item_id": product_info_name.get("data-item-id"),
                    "title": product_info_name.xpath(".//text()")[0],
                    "item_url": product_info_name.xpath("..")[0].get("href"),
                    "img_url": row.xpath(".//div[@class=\"__zoomable-thumbnail-inner\"]/img")[0].get("data-savepage-currentsrc"),
                    "avg_sold_price": row.xpath(".//td[@class=\"research-table-row__item research-table-row__avgSoldPrice\"]/div/div/text()")[0],
                    "avg_shipping_cost": row.xpath(".//td[@class=\"research-table-row__item research-table-row__avgShippingCost\"]/div/div/text()")[0],
                    "total_sold_count": row.xpath(".//td[@class=\"research-table-row__item research-table-row__totalSoldCount\"]/div/div/text()")[0],
                    "total_sales_value": row.xpath(".//td[@class=\"research-table-row__item research-table-row__totalSalesValue\"]/div/div/text()")[0],
                    "date_last_sold": row.xpath(".//td[@class=\"research-table-row__item research-table-row__dateLastSold\"]/div/div/text()")[0],
                })
            except Exception as e:
                print(f"EbayPageScraper: {e}")

        return pd.DataFrame(data)


if __name__ == "__main__":
    import os
    from src.common import utils

    df = EbayPageScraper.scrape_directory_to_dataframe()

    save_path = os.path.join(EbayPageScraper.BASE_SAVE_DIR, "ebay_seller_hub_scraped_tshirt_sales_data.pickle")
    utils.save_data(df, save_path)

    df = utils.load_data(save_path)

    print(df)
    print(df.iloc[0].loc["item_url"])
