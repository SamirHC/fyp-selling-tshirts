import pandas as pd
from PIL import Image, ImageFilter

from src.common import utils


def get_edges(img: Image.Image) -> Image.Image:
    """
    Applies edge detection filter on the input image.

    :param img: image to have edge detection applied to

    :return edges: image after applying edge detection and grayscale filters
    """

    return img.filter(ImageFilter.FIND_EDGES).convert("L")


def preprocessor(x: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses and extracts features from the raw input data from the
    dataframe for analysis.

    :param x: raw input dataframe

    :return x_pre: preprocessed dataframe
    """

    def remove_commas(s: str) -> str:
        return s.replace(",", "")

    x_pre = x.copy()

    x_pre["total_sold_count"] = (
        x_pre["total_sold_count"].apply(remove_commas).astype(int)
    )
    x_pre["total_sales_value"] = (
        x_pre["total_sales_value"]
        .apply(remove_commas)
        .apply(lambda s: str.strip(s, "£"))
        .astype(float)
    )
    x_pre["date_last_sold"] = pd.to_datetime(x_pre["date_last_sold"])

    x_pre["image"] = x["img_url"].apply(utils.get_image_from_url)
    x_pre["edges"] = x_pre["image"].apply(get_edges)

    x_pre = x_pre.drop(
        [
            "item_id",
            "item_url",
            "img_url",
            "avg_sold_price",
            "avg_shipping_cost",
        ],
        axis=1,
    )

    return x_pre


if __name__ == "__main__":
    import os
    from src.data_collection.ebay_page_scrape import EbayPageScraper

    path = os.path.join(EbayPageScraper.BASE_SAVE_DIR, "ebay_data.pickle")
    df = utils.load_data(path)

    print(df)

    x_pre = preprocessor(df)

    print(x_pre.dtypes)
    print(x_pre)

    for _, item in x_pre.iterrows():
        print(item)
        item["edges"].show()
