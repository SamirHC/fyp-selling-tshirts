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
        SELECT source, item_id FROM print_design_regions
        WHERE algorithm = 'SegformerB3ClothesSegmentation'
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
            c.source, c.item_id, c.title, c.image_url,
            pdr.left, pdr.top, pdr.width, pdr.height,
            pdr_inner_gt.left, pdr_inner_gt.top, pdr_inner_gt.width, pdr_inner_gt.height,
            pdr_outer_gt.left, pdr_outer_gt.top, pdr_outer_gt.width, pdr_outer_gt.height,
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
            ON c.source = pdr.source AND c.item_id = pdr.item_id AND pdr.algorithm = 'SegformerB3ClothesSegmentation'
        LEFT JOIN print_design_regions AS pdr_inner_gt
            ON c.source = pdr_inner_gt.source AND c.item_id = pdr_inner_gt.item_id AND pdr_inner_gt.algorithm = 'InnerGroundTruth'
        LEFT JOIN print_design_regions AS pdr_outer_gt
            ON c.source = pdr_outer_gt.source AND c.item_id = pdr_outer_gt.item_id AND pdr_outer_gt.algorithm = 'OuterGroundTruth'
        JOIN print_design_nearest_palette AS pdnp
            ON c.source = pdnp.source AND c.item_id = pdnp.design_id
        WHERE c.source = ? AND c.item_id = ?
        GROUP BY c.source, c.item_id
    """
    result = (
        source, item_id, title, image_url,
        left, top, width, height,
        in_left, in_top, in_width, in_height,
        out_left, out_top, out_width, out_height,
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

    # Draw bboxs on image
    image = utils.get_image_from_url(image_url).convert("RGB")
    draw = ImageDraw.ImageDraw(image)

    bbox = (left, top, left+width, top+height)
    draw.rectangle(bbox, outline=constants.Color.RED, width=3)  # Predicted

    if all((in_left, in_top, in_width, in_height)):  # Inner GT
        in_bbox = (in_left, in_top, in_left+in_width, in_top+in_height)
        draw.rectangle(in_bbox, outline=constants.Color.GREEN, width=3)
    
    if all((out_left, out_top, out_width, out_height)):  # Outer GT
        out_bbox = (out_left, out_top, out_left+out_width, out_top+out_height)
        draw.rectangle(out_bbox, outline=constants.Color.CYAN, width=3)

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
    source = f"Source: {source}\n"
    item_id = f"Item ID: {item_id}\n"
    title = f"Title: \n{textwrap.fill(title, width=30)} \n"
    tags = tags.split(",")
    tag_info = f"Assigned Tags:\n{"\n".join(tags)} \n"
    median = moving_median_df.loc[moving_median_df["x"]==palette_id, "log_y_smooth"].item()
    adjusted_popularity_score = f"Adjusted Popularity Score: {np.log(likes)/median}"
    likes = f"Likes: {likes} \n"
    palette_distance = f"Palette Distance: {palette_distance}\n"
    text_data = "\n".join(
        (source, item_id, title, tag_info, likes, adjusted_popularity_score, palette_distance)
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


def evaluate_palette_tag_matching(key, cursor: sqlite3.Cursor):
    while True:
        try:
            colour_match_count = int(input("Colour match count: "))
            tag_false_positive = int(input("Tag false positives: "))
            break
        except:
            print("Try again.")
    plt.close()
    print("-"*20)
    cursor.execute("""
        INSERT OR REPLACE INTO evaluate_palette_tag_matching (source, item_id, colour_match_count, tag_false_positives) 
        VALUES (?,?,?,?);
    """, (*key, colour_match_count, tag_false_positive))


if __name__ == "__main__":
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()

    keys = get_visualisable_clothes_keys(cursor)
    for key in keys:
        visualise_item_results(key, cursor)
        #evaluate_palette_tag_matching(key, cursor)
        #conn.commit()

    conn.close()
