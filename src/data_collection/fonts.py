import os
import sqlite3

import pandas as pd

from src.common import config


font_dir = os.path.join("data", "fonts", "dafonts-free-v1", "fonts")


def get_font_data_helper(query, params=None) -> pd.DataFrame:
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()

    result = cursor.execute(query, params)

    font_data = pd.DataFrame(
        result.fetchall(),
        columns=[d[0] for d in cursor.description]
    )
    font_data["path"] = font_data["filename"].apply(lambda filename: os.path.join(font_dir, filename))
    font_data = font_data.rename(columns={"base_font_name": "family", "file_format": "format"})

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


if __name__ == "__main__":
    print(get_font_data())
    print(get_font_data_by_family("Among Us"))
    print(get_random_fonts(10))
