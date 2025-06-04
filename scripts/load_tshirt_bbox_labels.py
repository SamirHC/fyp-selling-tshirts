import os
import sqlite3

import pandas as pd

from src.common import config


def load_bboxs(cursor: sqlite3.Cursor, df: pd.DataFrame):
    for _, row in df.iterrows():
        query = """
            SELECT source, item_id FROM clothes
            WHERE image_url=?
        """
        clothes_key = cursor.execute(query, (row["Mockup Image"],)).fetchone()
        
        query = """
            INSERT OR IGNORE INTO print_design_regions
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            *clothes_key,
            "InnerGroundTruth",
            row["x_small"], row["y_small"],
            row["width_small"], row["height_small"],
        )
        cursor.execute(query, params)

        query = """
            INSERT OR IGNORE INTO print_design_regions
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            *clothes_key,
            "OuterGroundTruth",
            row["x_big"], row["y_big"],
            row["width_big"], row["height_big"],
        )
        cursor.execute(query, params)
        conn.commit()


if __name__ == "__main__":
    tshirt_bbox_labels_path = os.path.join("data", "labels", "tshirt_bboxs.csv")
    df = pd.read_csv(tshirt_bbox_labels_path)
    
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()

    load_bboxs(conn, df)

    conn.close()
