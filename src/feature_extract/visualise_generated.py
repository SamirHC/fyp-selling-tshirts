import os
import sqlite3

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from src.common import config, utils


def get_df(cursor: sqlite3.Cursor):
    query = """
        SELECT * FROM evaluate_generations;
    """
    result = cursor.execute(query)
    columns = [d[0] for d in result.description]
    df = pd.DataFrame(result.fetchall(), columns=columns)

    return df


def get_reference_image(cursor: sqlite3.Cursor, clothes_key) -> Image.Image:
    query = """
        SELECT image_url FROM clothes
        WHERE source=? AND item_id=?
    """
    image_url = cursor.execute(query, clothes_key).fetchone()[0]
    image = utils.get_image_from_url(image_url).convert("RGB")
    return image


def get_generated_image(filename) -> Image.Image:
    path = os.path.join("data", "ea_population", filename)
    return Image.open(path).convert("RGBA")


def visualise(row: pd.Series, cursor: sqlite3.Cursor):
    clothes_key = row.loc["source"], row.loc["item_id"]
    ref_image = get_reference_image(cursor, clothes_key)
    gen_image = get_generated_image(row.loc["image_path"])

    fig, axs = plt.subplots(2,1)
    axs[0].imshow(ref_image)
    axs[1].imshow(gen_image)
    
    for i in range(2):
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    plt.show()


def main():
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()

    df = get_df(cursor)
    print(df)
    for i, row in df.iterrows():
        visualise(row, cursor)

    conn.close()


if __name__ == "__main__":
    main()
