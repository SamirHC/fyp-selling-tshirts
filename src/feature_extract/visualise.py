import sqlite3
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import skimage

from src.common import config, utils, constants
from src.data_collection import palettes
from src.ml.color_analysis.color_theme_classifier import CIELabColorThemeClassifier


moving_median_df = palettes.get_log_likes_moving_median(palettes.get_palette_data_db())


def get_visualisable_clothes_keys(cursor: sqlite3.Cursor) -> list:
    query = """
        SELECT source, item_id FROM clothes
        WHERE EXISTS (
            SELECT 1 FROM print_design_regions AS p
            WHERE clothes.source=p.source AND clothes.item_id=p.item_id
        )
        """
    return cursor.execute(query).fetchall()


def visualise_item_results(key: tuple, cursor: sqlite3.Cursor):
    """
    Given a clothes key from the SQL database, show the feature extraction
    pipeline results.
    """

    # Fetch data from db
    query = """
        SELECT
            c.title, c.image_url,
            pdr.left, pdr.top, pdr.width, pdr.height,
            (
                SELECT GROUP_CONCAT(colour)
                FROM print_design_palettes AS pdp
                WHERE pdp.source = c.source AND pdp.item_id = c.item_id
            ) AS print_design_palette,
            pdnp.palette_id,
            (
                SELECT GROUP_CONCAT(tag)
                FROM print_design_tags AS pdt
                WHERE pdt.source = c.source AND pdt.design_id = c.item_id
            ) AS tags
        FROM clothes AS c
        JOIN print_design_regions AS pdr
            ON c.source = pdr.source AND c.item_id = pdr.item_id
        JOIN print_design_nearest_palette AS pdnp
            ON c.source = pdnp.source AND c.item_id = pdnp.design_id
        WHERE c.source = ? AND c.item_id = ?
        GROUP BY c.source, c.item_id
    """
    result = (title, image_url, left, top, width, height,
     print_design_palette, palette_id, tags
    ) = cursor.execute(query, key).fetchone()
    print(result)

    query = """
        SELECT
            likes, color_hunt_id,
            (
                SELECT GROUP_CONCAT(colour) AS matched_palette
                FROM palette_colours
                WHERE palettes.id=palette_colours.palette_id
            )
        FROM palettes WHERE id=?
    """
    result = (likes, color_hunt_id, matched_palette
    ) = cursor.execute(query, (palette_id,)).fetchone()
    print(result)

    # Draw bbox on image
    image = utils.get_image_from_url(image_url).convert("RGB")
    bbox = (left, top, left+width, top+height)
    draw = ImageDraw.ImageDraw(image)
    draw.rectangle(bbox, outline=constants.Color.GREEN, width=3)

    # Format palettes
    print_design_palette = np.array(
        palettes.hex_palette_to_cielab_array(print_design_palette.split(","))
    ).reshape((-1, 3))
    matched_palette = np.array(
        palettes.hex_palette_to_cielab_array(matched_palette.split(","))
    ).reshape((-1, 3))
    palette_distance, matched_palette = CIELabColorThemeClassifier.palette_distance(print_design_palette, matched_palette)
    print_design_palette.resize((4, 1, 3))
    matched_palette.resize((4, 1, 3))
    print_design_palette = (skimage.color.lab2rgb(print_design_palette) * 255).astype(np.uint8)
    matched_palette = (skimage.color.lab2rgb(matched_palette) * 255).astype(np.uint8)

    # Other data
    title = f"Title: \n{textwrap.fill(title, width=30)} \n"
    tags = tags.split(",")
    tag_info = f"Assigned Tags: {"\n".join(tags)} \n"
    median = moving_median_df.loc[moving_median_df["x"]==palette_id, "log_y_smooth"].item()
    adjusted_popularity_score = f"Adjusted Popularity Score: {np.log(likes)/median}"
    likes = f"Likes: {likes} \n"
    palette_distance = f"Palette Distance: {palette_distance}\n"
    text_data = "\n".join(
        (title, tag_info, likes, adjusted_popularity_score, palette_distance)
    )

    # Create visualisation
    _, axs = plt.subplots(1, 4)

    axs[0].imshow(image)
    axs[1].imshow(print_design_palette)
    axs[1].set_title("Extracted Palette")
    axs[2].imshow(matched_palette)
    axs[2].set_title("Matched Palette")
    axs[3].text(0, 0.5, text_data, fontsize=12, va="center")

    for i in range(3):
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    axs[3].axis("off")

    plt.show()


if __name__ == "__main__":
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()

    keys = get_visualisable_clothes_keys(cursor)
    for key in keys:
        visualise_item_results(key, cursor)

    conn.close()
