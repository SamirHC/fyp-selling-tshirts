import os

import pandas as pd
import sqlite3

from src.data_collection.color_hunt_scrape import ColorHuntPageScraper
from src.data_collection.ebay_page_scrape import EbayPageScraper
from src.data_collection.etsy_page_scrape import EtsyPageScraper
from src.data_collection import ebay_browse


DB_PATH = os.path.join("data","db","dev_database.db")
SCHEMA_PATH = os.path.join("data","db","schema.sql")


def create_tables():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    with open(SCHEMA_PATH, "r") as file:
        script = file.read()
    cursor.executescript(script)
    conn.commit()
    conn.close()


def extract_data(**flags):
    default_flags = {
        "colorhunt": False,
        "ebay_browse": False,
        "ebay_seller_hub": False,
        "etsy": False,
    }
    flags = {key: flags.get(key, default) for key, default in default_flags.items()}

    extracted_data = {}

    if flags["colorhunt"]:
        colorhunt_html_path = ColorHuntPageScraper.download_html()
        colorhunt_df = ColorHuntPageScraper.scrape_html_to_dataframe(colorhunt_html_path)
        if len(colorhunt_df):
            extracted_data["colorhunt"] = colorhunt_df

    if flags["ebay_seller_hub"]:
        ebay_seller_hub_df = EbayPageScraper.scrape_directory_to_dataframe()
        if len(ebay_seller_hub_df):
            extracted_data["ebay_seller_hub"] = ebay_seller_hub_df
    
    if flags["ebay_browse"]:
        access_token = ebay_browse.get_access_token()
        queries = ["funny graphic tee", "inspirational tshirt"]
        ebay_browse_df = pd.concat(
            ebay_browse.get_items_as_dataframe(access_token, query)
            for query in queries
        )
        if len(ebay_browse_df):
            extracted_data["ebay_browse"] = ebay_browse_df
    
    if flags["etsy"]:
        etsy_df = EtsyPageScraper.scrape_directory_to_dataframe()
        if len(etsy_df):
            extracted_data["etsy"] = etsy_df

    return extracted_data


def transform_data(data):
    transformed_data = {}

    if "colorhunt" in data:
        colorhunt_df: pd.DataFrame = data["colorhunt"]
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

        transformed_data |= {
            "palette_colours": palette_colours_df,
            "palette_tag_associations": palette_tag_associations_df,
            "palette_tags": palette_tags_df,
            "palettes": palettes_df,
        }

    clothes_df = pd.DataFrame()

    if "ebay_seller_hub" in data:
        ebay_seller_hub_df: pd.DataFrame = data["ebay_seller_hub"]
        ebay_seller_hub_clothes_df = ebay_seller_hub_df[["item_id", "title", "img_url"]].dropna()
        ebay_seller_hub_clothes_df.rename(columns={"img_url": "image_url"}, inplace=True)
        ebay_seller_hub_clothes_df["source"] = "Ebay Seller Hub"
        clothes_df = pd.concat([clothes_df, ebay_seller_hub_clothes_df])

    if "ebay_browse" in data:
        ebay_browse_df: pd.DataFrame = data["ebay_browse"]
        ebay_browse_clothes_df = ebay_browse_df[["item_id", "title", "img_url"]].dropna()
        ebay_browse_clothes_df.rename(columns={"img_url": "image_url"}, inplace=True)
        ebay_browse_clothes_df["source"] = "Ebay Browse API"
        clothes_df = pd.concat([clothes_df, ebay_browse_clothes_df])

    if "etsy" in data:
        etsy_df: pd.DataFrame = data["etsy"]
        etsy_clothes_df = etsy_df[["item_id", "title", "img_url"]].dropna()
        etsy_clothes_df.rename(columns={"img_url": "image_url"}, inplace=True)
        etsy_clothes_df["source"] = "Etsy Listing"
        clothes_df = pd.concat([clothes_df, etsy_clothes_df])

    if len(clothes_df):
        transformed_data["clothes"] = clothes_df

    return transformed_data


def load_data(data):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if "palettes" in data:
        palette_tags_df: pd.DataFrame = data["palette_tags"]
        cursor.executemany("""
            INSERT OR IGNORE INTO palette_tags (name, is_colour_tag)
            VALUES (?, ?)
        """, palette_tags_df.values.tolist())

        palettes_df: pd.DataFrame = data["palettes"]
        cursor.executemany("""
            INSERT INTO palettes (id, likes, submission_date, color_hunt_id)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (color_hunt_id)
            DO UPDATE SET likes=excluded.likes, submission_date=excluded.submission_date
        """, palettes_df.values.tolist())

        palette_colours_df: pd.DataFrame = data["palette_colours"]
        cursor.executemany("""
            INSERT OR IGNORE INTO palette_colours (palette_id, colour)
            VALUES (?, ?)
        """, palette_colours_df.values.tolist())

        palette_tag_associations_df: pd.DataFrame = data["palette_tag_associations"]
        cursor.executemany("""
            INSERT OR IGNORE INTO palette_tag_associations (palette_id, tag)
            VALUES (?, ?)
        """, palette_tag_associations_df.values.tolist())

        conn.commit()

    if "clothes" in data:
        clothes_df: pd.DataFrame = data["clothes"]
        cursor.executemany("""
            INSERT OR IGNORE INTO clothes (item_id, title, image_url, source)
            VALUES (?, ?, ?, ?)
        """, clothes_df.values.tolist())
        conn.commit()

    conn.close()


def etl_pipeline(**flags):
    """
    flags: colorhunt, etsy, ebay_seller_hub, ebay_browse
    """

    create_tables()
    data = extract_data(**flags)
    data = transform_data(data)
    load_data(data)


if __name__ == "__main__":
    etl_pipeline(colorhunt=False, etsy=False, ebay_seller_hub=False, ebay_browse=False)
