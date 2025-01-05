import os

import pandas as pd
import pytest

from src.tshirt_data_collection import ebay_page_scrape


@pytest.fixture(scope="session")
def df():
    path = os.path.join("test", "test_data", "raw", "ebay_seller_hub_research_html_pages", "Product Research - Seller Hub.html")
    return ebay_page_scrape.scrape_to_dataframe(path)


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
    ebay_page_scrape.save_data(df, save_path=save_path)
    loaded_df = ebay_page_scrape.load_data(save_path)

    pd.testing.assert_frame_equal(df, loaded_df)


if __name__ == "__main__":
    import sys
    pytest.main(sys.argv)
