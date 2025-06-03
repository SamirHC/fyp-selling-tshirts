import pandas as pd
from PIL import Image

from src.data_collection import fonts


# Determine font based on prompt and image data. Currently random.
def select_font_tags(prompt: str, image: Image.Image) -> tuple:
    categories = list(fonts.get_category_names())
    themes = list(fonts.get_theme_names())
    categories.remove("Dingbats")
    category = pd.Series(categories).sample(n=1).item()
    theme = pd.Series(themes).sample(n=1).item()
    return category, theme


def select_font(prompt: str, image: Image.Image) -> str:
    font_df = fonts.get_font_data()
    font_df = font_df.loc[font_df["category"]!="Dingbats"]
    category, theme = select_font_tags(prompt, image)
    
    font_theme_df: pd.DataFrame = font_df.loc[font_df["theme"]==theme]
    if len(font_theme_df):
        font_theme_category_df: pd.DataFrame = font_theme_df.loc[font_theme_df["category"]==category]
        if len(font_theme_category_df):
            df = font_theme_category_df
            print("Font theme and category selected")
        else:
            df = font_theme_df
            print("Font theme selected only")
    else:
        df = font_df
        print("Neither font theme nor category selected")

    return df.sample(n=1)["family"].item()


if __name__ == "__main__":
    tags = select_font_tags(None, None)
    print(tags)
    font = select_font("djsk", None)
    print(font)
