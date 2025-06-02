import argparse
import sqlite3

from src.common import utils, config
from src.data_collection import palettes
from src.ml.color_analysis.color_theme_classifier import CIELabColorThemeClassifier
from src.ml.tshirt_design_segmentation import segmentation


def find_clothes_to_feature_extract(cursor: sqlite3.Cursor) -> list:
    query = """
        SELECT source, item_id FROM clothes
        WHERE NOT EXISTS (
            SELECT 1 FROM print_design_regions AS p
            WHERE clothes.source=p.source AND clothes.item_id=p.item_id
        )"""
    return cursor.execute(query).fetchall()


def clear_features() -> list:
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()

    query = """ 
        DELETE FROM print_design_nearest_palette;
        DELETE FROM print_design_palettes;
        DELETE FROM print_design_regions;
        DELETE FROM print_design_tags;
    """
    cursor.executescript(query)

    conn.commit()
    conn.close()


def get_clothes_img_url(cursor: sqlite3.Cursor, clothes_key: tuple[str, str]) -> str:
    query = "SELECT image_url FROM clothes WHERE source=? AND item_id=?"
    return cursor.execute(query, clothes_key).fetchone()


def load_features(n=-1):
    segmentation_model = segmentation.SegformerB3ClothesSegmentation()

    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()

    clothes_keys = find_clothes_to_feature_extract(cursor)
    while clothes_keys and n != 0:
        n -= 1
        key = clothes_keys.pop()
        print(key)
        img_url = get_clothes_img_url(cursor, key)[0]
        image = utils.get_image_from_url(img_url).convert("RGB")

        x1, y1, x2, y2 = segmentation_model.extract_design_bbox(image)
        w = x2 - x1
        h = y2 - y1
        design_image = segmentation_model.extract_design(image)

        palette_data = CIELabColorThemeClassifier.get_palette_data(design_image)
        tags = palette_data["color_tags"] + palette_data["other_tags"]
        colours = [palettes.rgb_to_hex(map(int, colour)).upper() for colour in palette_data["colors"]]

        query = """
            INSERT OR IGNORE INTO print_design_regions (source, item_id, algorithm, left, top, width, height)
            VALUES (?,?,?,?,?,?,?)
        """
        cursor.execute(query, (key[0], key[1], segmentation_model.__class__.__name__, int(x1), int(y1), int(w), int(h)))
        
        query = """
            INSERT OR IGNORE INTO print_design_tags (source, design_id, tag)
            VALUES (?, ?, ?)
        """
        cursor.executemany(query, ((key[0], key[1], tag) for tag in tags))

        query = """
            INSERT OR IGNORE INTO print_design_palettes (source, item_id, colour)
            VALUES (?, ?, ?)
        """
        cursor.executemany(query, ((key[0], key[1], colour) for colour in colours))

        query = """
            INSERT OR IGNORE INTO print_design_nearest_palette (source, design_id, palette_id, distance)
            VALUES (?, ?, ?, ?)
        """
        cursor.execute(query, (key[0], key[1], palette_data["row_idx"], palette_data["dist"]))

        conn.commit()

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Feature extraction")
    parser.add_argument("--clear", action="store_true", help="Clear extracted features")
    parser.add_argument("--load", action="store_true", help="Load features")
    parser.add_argument("--n", type=int, default=-1, help="Limit number of items to load")

    args = parser.parse_args()

    if args.clear:
        print("Clearing feature table data")
        clear_features()
    if args.load:
        print("Loading features")
        load_features(n=args.n)


if __name__ == "__main__":
    main()
