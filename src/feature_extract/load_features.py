import os
import sqlite3

import pandas as pd

from src.common import utils
from src.ml.color_analysis.color_theme_classifier import CIELabColorThemeClassifier
from src.ml.tshirt_design_segmentation import segmentation


DB_PATH = os.path.join("data","db","dev_database.db")


def find_clothes_to_feature_extract(cursor: sqlite3.Cursor) -> list:
    query = """
        SELECT source, item_id FROM clothes
        WHERE NOT EXISTS (
            SELECT 1 FROM palette_distances AS p
            WHERE clothes.source=p.source AND clothes.item_id=p.design_id
        )"""
    return cursor.execute(query).fetchall()


def get_clothes_img_url(cursor: sqlite3.Cursor, clothes_key: tuple[str, str]) -> str:
    query = "SELECT img_url FROM clothes WHERE source=? AND item_id=?"
    return cursor.execute(query, clothes_key).fetchone()


def load_features(segmentation_model: segmentation.TshirtDesignSegmentationModel):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    clothes_keys = find_clothes_to_feature_extract(cursor)
    if clothes_keys:
        key = clothes_keys[0]
        img_url = get_clothes_img_url(cursor, key)
        image = utils.get_image_from_url(img_url).convert("RGB")

        design_image = segmentation_model.extract_design(image)
        
        palette_data = CIELabColorThemeClassifier.get_palette_data(design_image)
        palette_data["color_tags"]
        palette_data["other_tags"]
        palette_data["colors"]


    conn.close()

    print(clothes_keys)


def extract_design_data(df: pd.DataFrame, **kwargs):
    segmentation_model: segmentation.TshirtDesignSegmentationModel = (
        kwargs.get("tshirt_design_segmentation_model", segmentation.ProceduralSegmentation())
    )

    def _analyse_image(url):
        image = utils.get_image_from_url(url).convert("RGB")
        design_image = segmentation_model.extract_design(image)

        insights = {}

        palette_data = CIELabColorThemeClassifier.get_palette_data(design_image)
        insights["color_tags"] = palette_data["color_tags"]
        insights["other_tags"] = palette_data["other_tags"]
        insights["colors"] = palette_data["colors"]

        # TODO: 
        #  - Resnet Image Classification or other: get key nouns in image and
        #    and collect for semantic data
        #  - Classify design as text, image, both or neither

        return pd.Series(insights)

    df[["color_tags", "other_tags", "colors"]] = df["img_url"].apply(_analyse_image)


if __name__ == "__main__":
    load_features()
