from lxml import etree
import pandas as pd
import pickle
import os


def scrape_to_dataframe(src: str) -> pd.DataFrame:
    with open(src, "r") as f:
        tree = etree.HTML(f.read())
        rows = tree.xpath("//table/tr[contains(@class, \"research\")]")[1:]

    data = []

    for row in rows:
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

    return pd.DataFrame(data)


def save_data(df: pd.DataFrame):
    with open(os.path.join("data", "ebay_data.pickle"), "wb") as target:
        pickle.dump(df, target)
    print("Saved data to ebay_data.pickle")


def load_data():
    with open(os.path.join("data", "ebay_data.pickle"), "rb") as target:
        df = pickle.load(target)
    print("Loaded data from ebay_data.pickle\n")
    return df


if __name__ == "__main__":
    import os

    filepath = os.path.join("ebay_pages", "Product Research - Seller Hub.html")

    save_data(scrape_to_dataframe(filepath))
    df = load_data()
    print(df)
    print(df.iloc[0].loc["item_url"])
