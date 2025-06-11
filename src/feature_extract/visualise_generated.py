import os
import sqlite3
import textwrap

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from src.common import config, utils
from src.data_collection import palettes


moving_median_df = palettes.get_log_likes_moving_median(palettes.get_palette_data_db())


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


def get_reference_title(cursor: sqlite3.Cursor, clothes_key) -> str:
    query = """
        SELECT title FROM clothes
        WHERE source=? AND item_id=?
    """
    title = cursor.execute(query, clothes_key).fetchone()[0]
    return title


def visualise(row: pd.Series, cursor: sqlite3.Cursor):
    clothes_key = row.loc["source"], row.loc["item_id"]

    # Images
    ref_image = get_reference_image(cursor, clothes_key)
    gen_image = get_generated_image(row.loc["image_path"])

    # Palette metrics
    palette_id = row["palette_id"]
    palette_data = palettes.get_palette_data_by_id(cursor, palette_id)
    likes = palette_data["likes"]
    median = moving_median_df.loc[moving_median_df["x"]==palette_id, "log_y_smooth"].item()
    adjusted_popularity_score = f"Adjusted Popularity Score: {np.log(likes)/median}"
    nearest_colours = palettes.hex_palette_to_rgb_array(palette_data["colors"])
    nearest_colours = np.array(nearest_colours).reshape(1, -1, 3)

    ref_text_data = textwrap.fill(get_reference_title(cursor, clothes_key), width=30)
    gen_text_data = textwrap.fill(row["prompt"], width=30)

    NUM_ROWS = 2
    NUM_COLS = 3
    fig, axs = plt.subplots(NUM_ROWS, NUM_COLS)
    axs[0][0].text(0, 0.5, ref_text_data, fontsize=12, va="center")
    axs[1][0].text(0, 0.5, gen_text_data, fontsize=12, va="center")
    
    axs[0][1].imshow(ref_image)
    axs[1][1].imshow(gen_image)


    for i in range(NUM_ROWS):
        for j in range(NUM_COLS):
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])

    plt.show()


def show_score_hist(cursor: sqlite3.Cursor):
    scores = cursor.execute(
        """SELECT colour_score, prompt_score, aesthetic_score FROM evaluate_generations"""
    ).fetchall()
    colour_scores, prompt_scores, aesthetic_scores = zip(*scores)
    
    fig, axs = plt.subplots(1, 3)
    axs[0].hist(colour_scores, bins=6)
    axs[0].set_xlabel("Colour Score")
    axs[0].set_ylabel("Frequency")
    axs[1].hist(prompt_scores, bins=6)
    axs[1].set_xlabel("Prompt Score")
    axs[2].hist(aesthetic_scores, bins=6)
    axs[2].set_xlabel("Aesthetic Score")

    plt.show()

    print(f"colour_score: mean={np.mean(colour_scores)}")
    print(f"prompt_scores: mean={np.mean(prompt_scores)}")
    print(f"aesthetic_scores: mean={np.mean(aesthetic_scores)}")


def main():
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()

    show_score_hist(cursor)

    df = get_df(cursor)
    print(df)
    for i, row in df.iterrows():
        visualise(row, cursor)

    conn.close()


if __name__ == "__main__":
    main()
