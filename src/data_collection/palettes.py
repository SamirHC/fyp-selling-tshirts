import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage

from src.common import utils


BASE_DIR = os.path.join("data", "dataframes", "color_hunt_palette_data")


def get_palette_data() -> pd.DataFrame:
    file_path = os.path.join(BASE_DIR, "ColorHuntPageScraper 2025-01-08 11:51:25.pickle")
    return utils.load_data(file_path)


def hex_to_rgb(hex: str) -> tuple[int, int, int]:
    hex = hex.lstrip('#')
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def hex_palette_to_rgb_array(palette: list[str]) -> np.ndarray:
    return np.array([hex_to_rgb(hex) for hex in palette], dtype=np.uint8)


def hex_palette_to_cielab_array(palette: list[str]) -> np.ndarray:
    return skimage.color.rgb2lab(hex_palette_to_rgb_array(palette) / 255)


def rgb_array_palette_to_cielab_array(palette: np.ndarray) -> np.ndarray:
    return skimage.color.rgb2lab(palette / 255)


def get_tags(palette_data: pd.DataFrame) -> set[str]:
    return set((palette_data["color_tags"] + palette_data["other_tags"]).explode().unique())


def get_color_tag_counts(palette_data: pd.DataFrame) -> pd.Series:
    return palette_data.explode("color_tags")["color_tags"].value_counts()


def get_other_tag_counts(palette_data: pd.DataFrame) -> pd.Series:
    return palette_data.explode("other_tags")["other_tags"].value_counts()


def show_likes(palette_data: pd.DataFrame):
    likes_df = palette_data["likes"].apply(lambda x: int(x.replace(",", "")))
    plt.scatter(likes_df.index, likes_df.iloc[::-1], s=10, marker="x")
    plt.yscale("log")
    plt.xlabel("Nth Palette Submission to Color Hunt")
    plt.ylabel("Number of Likes (Log Scale)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    palette_data = get_palette_data()

    print([hex_to_rgb(hex) for hex in palette_data.iloc[0]["colors"]])
    print(get_tags(palette_data))
    print(palette_data.iloc[0]["url"])

    color_tag_counts = get_color_tag_counts(palette_data)
    other_tag_counts = get_other_tag_counts(palette_data)

    print(color_tag_counts)
    print(other_tag_counts)

    show_likes(palette_data)

    #plt.pie(color_tag_counts)
    #plt.show()
