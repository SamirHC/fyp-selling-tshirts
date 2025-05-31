import os
import sqlite3

import pandas as pd

from src.common import config


font_dir = os.path.join("data", "fonts", "dafonts-free-v1", "fonts")


def get_font_data(limit=100000, offset=0) -> pd.DataFrame:
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()

    result = cursor.execute("""
        SELECT * FROM fonts LIMIT ? OFFSET ?
    """, (limit, offset))

    font_data = pd.DataFrame(
        result.fetchall(),
        columns=[d[0] for d in cursor.description]
    )
    font_data["path"] = font_data["filename"].apply(lambda filename: os.path.join(font_dir, filename))
    font_data = font_data.rename(columns={"base_font_name": "family", "file_format": "format"})

    conn.close()
    return font_data


if __name__ == "__main__":
    print(get_font_data())
