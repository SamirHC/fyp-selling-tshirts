import os

import pandas as pd
import sqlite3

from src.data_collection import palettes
from src.data_collection.color_hunt_scrape import ColorHuntPageScraper


def extract_data():
    ### Download new content
    #colorhunt_html_path = ColorHuntPageScraper.download_html(3)

    ### Extract desired data to df
    #colorhunt_df = ColorHuntPageScraper.scrape_html_to_dataframe(colorhunt_html_path)
    #print(colorhunt_df)

    # Placeholder data
    colorhunt_df = palettes.get_palette_data()

    return {"colorhunt": colorhunt_df}


def transform_data(data):
    ### Palettes
    colorhunt_df: pd.DataFrame = data["colorhunt"][::-1]
    colorhunt_df.reset_index(drop=True, inplace=True)
    colorhunt_df.reset_index(inplace=True)

    palettes_df = colorhunt_df[["index", "likes", "date", "palette_id"]]
    palettes_df.columns = ("id", "likes", "submission_date", "color_hunt_id")

    color_tags_df = colorhunt_df[["index", "color_tags"]].explode("color_tags").dropna().drop_duplicates()
    other_tags_df = colorhunt_df[["index", "other_tags"]].explode("other_tags").dropna().drop_duplicates()
    color_tags_df["is_colour_tag"] = True
    other_tags_df["is_colour_tag"] = False
    color_tags_df.rename(columns={"color_tags": "name"}, inplace=True)
    other_tags_df.rename(columns={"other_tags": "name"}, inplace=True)
    all_tags_df = pd.concat((color_tags_df, other_tags_df))

    palette_tag_associations_df = all_tags_df[["index", "name"]]
    palette_tag_associations_df.columns = ("palette_id", "tag")

    palette_tags_df = all_tags_df[["name", "is_colour_tag"]].drop_duplicates().reset_index(drop=True)

    palette_colours_df = colorhunt_df[["index", "colors"]].explode("colors")
    palette_colours_df.columns = ("palette_id", "colour")

    ### Clothes
    clothes_df = None
    clothes_sources_df = None

    ### Print Design Features
    print_design_palettes_df = None
    print_design_regions_df = None
    palette_distances_df = None

    transformed_data = {
        "clothes": clothes_df,
        "clothes_sources": clothes_sources_df,
        "palette_colours": palette_colours_df,
        "palette_distances": palette_distances_df,
        "palette_tag_associations": palette_tag_associations_df,
        "palette_tags": palette_tags_df,
        "palettes": palettes_df,
        "print_design_palettes": print_design_palettes_df,
        "print_design_regions": print_design_regions_df,
    }
    return transformed_data


def load_data(data):
    DB_PATH = os.path.join("data","db","dev_database.db")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    SCHEMA_PATH = os.path.join("data","db","schema.sql")
    with open(SCHEMA_PATH, "r") as file:
        script = file.read()
    cursor.executescript(script)

    palette_tags_df: pd.DataFrame = data["palette_tags"]
    palette_tags_df.to_sql("palette_tags", conn, if_exists="append", index=False)

    palettes_df: pd.DataFrame = data["palettes"]
    cursor.executemany("""
        INSERT INTO "palettes" ("id", "likes", "submission_date", "color_hunt_id")
        VALUES (?, ?, ?, ?)
        ON CONFLICT ("color_hunt_id")
        DO UPDATE SET "likes" = excluded."likes", "submission_date" = excluded."submission_date"
    """, palettes_df.values.tolist())

    palette_colours_df: pd.DataFrame = data["palette_colours"]
    palette_colours_df.to_sql("palette_colours", conn, if_exists="append", index=False)

    palette_tag_associations_df: pd.DataFrame = data["palette_tag_associations"]
    palette_tag_associations_df.to_sql("palette_tag_associations", conn, if_exists="append", index=False)

    conn.commit()
    conn.close()


def etl_pipeline():
    data = extract_data()
    data = transform_data(data)
    load_data(data)


if __name__ == "__main__":
    etl_pipeline()
