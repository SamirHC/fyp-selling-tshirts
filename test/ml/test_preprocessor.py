import os
import pickle

import pandas as pd
from PIL import Image
import pytest

from src.common import utils
from src.ml import preprocessor


@pytest.fixture(scope="session")
def df():
    path = os.path.join("test", "test_data", "dataframes", "seller_hub_data", "ebay_data.pickle")
    
    with open(path, "rb") as target:
        df = pickle.load(target)

    return df[0:10]


@pytest.fixture(scope="session")
def image(df):
    row = df.iloc[0]
    url = row["img_url"]

    return utils.get_image_from_url(url)


def test_get_image(image):
    assert isinstance(image, Image.Image)


def test_get_edges(image):
    edges = preprocessor.get_edges(image)

    assert isinstance(edges, Image.Image)
    assert edges.size == image.size


def test_preprocessor(df):
    x_pre = preprocessor.preprocessor(df)
    expected_columns = pd.Series({
        "title": object,
        "total_sold_count": int,
        "total_sales_value": float,
        "date_last_sold": "datetime64[ns]",
        "image": object,
        "edges": object,
    })

    assert x_pre.shape == (10, len(expected_columns))
    assert (x_pre.dtypes == expected_columns).all()


if __name__ == "__main__":
    import sys
    pytest.main(sys.argv)
