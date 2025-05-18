import os
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage

from src.common import utils


BASE_DIR = os.path.join("data", "dataframes", "color_hunt_palette_data")
DB_PATH = os.path.join("data","db","dev_database.db")


def get_palette_data() -> pd.DataFrame:
    file_path = os.path.join(BASE_DIR, "ColorHuntPageScraper 2025-01-08 11:51:25.pickle")
    return utils.load_data(file_path)


def iterate_through_palette_db(cursor: sqlite3.Cursor):
    palette_id = 0
    while True:
        main_query = "SELECT likes, submission_date, color_hunt_id FROM palettes WHERE id=?"
        result = cursor.execute(main_query, (palette_id,)).fetchone()
        if result is None:
            break
        likes, submission_date, color_hunt_id = result

        colour_query = "SELECT colour FROM palette_colours WHERE palette_id=?"
        colours = [x[0] for x in cursor.execute(colour_query, (palette_id,)).fetchall()]

        tag_query = """
            SELECT tag FROM palette_tag_associations
                JOIN palette_tags ON palette_tag_associations.tag=palette_tags.name
            WHERE palette_id=? AND is_colour_tag=?
        """
        colour_tags = [x[0] for x in cursor.execute(tag_query, (palette_id, 1)).fetchall()]
        other_tags = [x[0] for x in cursor.execute(tag_query, (palette_id, 0)).fetchall()]

        yield pd.Series({
            "palette_id": palette_id,
            "colors": colours,
            "likes": likes,
            "color_tags": colour_tags,
            "other_tags": other_tags,
            "date": submission_date,
            "url": f"https://colorhunt.co/palette/{color_hunt_id}"
        })

        palette_id += 1


def get_palette_data_db() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    df = pd.DataFrame(iterate_through_palette_db(cursor))

    conn.close()
    return df


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
    likes_df = palette_data["likes"]
    plt.scatter(likes_df.index, likes_df.iloc[::-1], s=10, marker="x")
    plt.yscale("log")
    plt.xlabel("Nth Palette Submission to Color Hunt")
    plt.ylabel("Number of Likes (Log Scale)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    palette_data = get_palette_data_db()
    print(palette_data)
    likes = palette_data.iloc[0]["likes"]
    print(likes, type(likes))

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
