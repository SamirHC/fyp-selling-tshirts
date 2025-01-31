import os

import pandas as pd
import pytest

from src.data_collection.ebay_page_scrape import EbayPageScraper
from src.common import utils


@pytest.fixture(scope="session")
def df():
    path = os.path.join("test", "test_data", "raw", "ebay_seller_hub_research_html_pages", "Product Research - Seller Hub.html")
    return EbayPageScraper.scrape_html_to_dataframe(path)


def test_scrape_to_dataframe_returns_dataframe(df):
    assert isinstance(df, pd.DataFrame)


def test_scrape_to_dataframe_shape(df):
    assert df.shape == (50, 9)


def test_scrape_to_dataframe_columns(df):
    expected_columns = [
        "item_id",
        "title",
        "item_url",
        "img_url",
        "avg_sold_price",
        "avg_shipping_cost",
        "total_sold_count",
        "total_sales_value",
        "date_last_sold"
    ]
    actual_columns = df.columns

    for c in expected_columns:
        assert c in actual_columns


def test_scrape_to_dataframe_values(df):
    assert df.iloc[0].loc["item_url"] == "https://www.ebay.co.uk/itm/176168653540?nordt=true&orig_cvip=true&rt=nc"


def test_save_data(df, tmp_path):
    save_path = tmp_path / "saved_test_seller_hub_data.pickle"
    utils.save_data(df, save_path=save_path)
    loaded_df = utils.load_data(save_path)

    pd.testing.assert_frame_equal(df, loaded_df)


if __name__ == "__main__":
    import sys
    pytest.main(sys.argv)
