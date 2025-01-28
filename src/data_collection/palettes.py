import os

import pandas as pd

from src.common import utils


BASE_DIR = os.path.join("data", "dataframes", "color_hunt_palette_data")


def get_palette_data() -> pd.DataFrame:
    file_path = os.path.join(BASE_DIR, "ColorHuntPageScraper 2025-01-08 11:51:25.pickle")
    return utils.load_data(file_path)

def hex_to_rgb(hex: str) -> tuple[int, int, int]:
    hex = hex.lstrip('#')
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

if __name__ == "__main__":
    palette_data = get_palette_data()

    print([hex_to_rgb(hex) for hex in palette_data.iloc[0]["colors"]])
