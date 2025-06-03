from functools import lru_cache
import os
import sqlite3

import pandas as pd

from src.common import config


FONT_DIR = os.path.join("data", "fonts", "dafonts-free-v1", "fonts")


def get_font_path(filename):
    return os.path.join(FONT_DIR, filename)


def get_font_data_helper(query, params=None) -> pd.DataFrame:
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()

    result = cursor.execute(query, params)

    font_data = pd.DataFrame(
        result.fetchall(),
        columns=[d[0] for d in cursor.description]
    )
    font_data["path"] = font_data["filename"].apply(get_font_path)
    font_data = font_data.rename(columns={
        "base_font_name": "family",
        "file_format": "format"
    })

    conn.close()
    return font_data


def get_font_data(limit=100000, offset=0) -> pd.DataFrame:
    query = """
        SELECT * FROM fonts LIMIT ? OFFSET ?
    """
    params = limit, offset
    return get_font_data_helper(query, params)


def get_font_data_by_family(family) -> pd.DataFrame:
    query = """
        SELECT * FROM fonts WHERE base_font_name=?
    """
    params = (family, )
    return get_font_data_helper(query, params)


def get_random_fonts(limit=1) -> pd.DataFrame:
    query = """
        SELECT * FROM fonts ORDER BY RANDOM() LIMIT ?
    """
    params = (limit,)
    return get_font_data_helper(query, params)


def get_themes() -> pd.DataFrame:
    query = """
        SELECT DISTINCT theme, COUNT(theme) AS count
        FROM fonts
        GROUP BY theme
    """

    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    themes = cursor.execute(query).fetchall()
    columns = [d[0] for d in cursor.description]
    conn.close()

    return pd.DataFrame(themes, columns=columns)


def get_categories() -> pd.DataFrame:
    query = """
        SELECT DISTINCT category, COUNT(category) AS count
        FROM fonts
        GROUP BY category
    """

    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    categories = cursor.execute(query).fetchall()
    columns = [d[0] for d in cursor.description]
    conn.close()

    return pd.DataFrame(categories, columns=columns)


@lru_cache(maxsize=1)
def get_theme_names():
    return tuple(get_themes()["theme"].to_list())

@lru_cache(maxsize=1)
def get_category_names():
    return tuple(get_categories()["category"].to_list())


if __name__ == "__main__":
    print(get_font_data())
    print(get_font_data_by_family("Among Us"))
    print(get_random_fonts(10))
    print(get_themes())
    print(get_categories())
    print(get_theme_names())
    print(get_category_names())
