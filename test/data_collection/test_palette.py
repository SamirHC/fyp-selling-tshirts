import os

import pandas as pd
import pytest

from src.data_collection import palettes
from src.common import utils


@pytest.fixture(scope="session")
def df():
    path = os.path.join("test", "test_data", "dataframes", "color_hunt_palette_data", "ColorHuntPageScraper 2025-01-08 11:51:25.pickle")
    return utils.load_data(path)


def test_get_palette_data_cols():
    expected_columns = [
        "palette_id",
        "colors",
        "likes",
        "color_tags",
        "other_tags",
        "date",
        "url",
    ]

    actual_columns = palettes.get_palette_data().columns

    assert set(expected_columns) == set(actual_columns)


def test_hex_to_rgb():
    assert palettes.hex_to_rgb("#000000") == (0, 0, 0)
    assert palettes.hex_to_rgb("#ffffff") == (255, 255, 255)
    assert palettes.hex_to_rgb("#ff0000") == (255, 0, 0)
    assert palettes.hex_to_rgb("#00ff00") == (0, 255, 0)
    assert palettes.hex_to_rgb("#0000ff") == (0, 0, 255)
    assert palettes.hex_to_rgb("#ab023e") == (171, 2, 62)


def test_get_tags(df):
    tags = palettes.get_tags(df)
    assert isinstance(tags, set)

    for tag in tags:
        assert isinstance(tag, str)
