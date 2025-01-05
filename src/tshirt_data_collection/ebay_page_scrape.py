import os
import pickle

from lxml import etree
import pandas as pd


SELLER_HUB_DATA_DIR = os.path.join("data", "dataframes", "seller_hub_data")
DEFAULT_SELLER_HUB_DATA_PATH = os.path.join(SELLER_HUB_DATA_DIR, "ebay_data.pickle")


def scrape_to_dataframe(src: str) -> pd.DataFrame:
    """
    Extracts the item sales data from an eBay Seller Hub Research HTML page.

    :param src: eBay Seller Hub Research HTML page file path

    :return df: dataframe of extracted item sales data
    """

    with open(src, "r") as f:
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
            print(e)

    return pd.DataFrame(data)


def save_data(df: pd.DataFrame, save_path=None):
    """
    Saves the dataframe to a pickle file.
    
    :param df: the dataframe to save
    :param save_path: the optional file path to save to (default 
    ebay_data.pickle)
    """

    if save_path is None:
        save_path = DEFAULT_SELLER_HUB_DATA_PATH

    with open(save_path, "wb") as target:
        pickle.dump(df, target)

    print(f"Saved data to {save_path}")


def load_data(path=None) -> pd.DataFrame:
    """
    Loads the pickle file into a dataframe.

    :param path: optional path to load from (default ebay_data.pickle)

    :return df: the dataframe contents of the pickle file
    """

    if path is None:
        path = DEFAULT_SELLER_HUB_DATA_PATH

    with open(path, "rb") as target:
        df = pickle.load(target)

    print(f"Loaded data from {path}\n")
    return df


if __name__ == "__main__":
    import os

    BASE_DIR = os.path.join("data", "raw", "ebay_seller_hub_research_html_pages")
    save_path = os.path.join(SELLER_HUB_DATA_DIR, "ebay_seller_hub_scraped_tshirt_sales_data.pickle")

    dfs = []
    for filepath in os.listdir(BASE_DIR):
        abspath = os.path.join(BASE_DIR, filepath)

        df = scrape_to_dataframe(abspath)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)

    save_data(df, save_path=save_path)
    df = load_data(save_path)
    print(df)
    print(df.iloc[0].loc["item_url"])
