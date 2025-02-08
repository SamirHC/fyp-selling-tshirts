import os
from enum import Enum

import pandas as pd


class Language(Enum):
    ENGLISH = "en"
    JAPANESE = "jp"


FONT_PATH = os.path.join("data", "fonts")
EN_FONT_PATH = os.path.join(FONT_PATH, Language.ENGLISH.value)
JP_FONT_PATH = os.path.join(FONT_PATH, Language.JAPANESE.value)


def get_font_data(lang=Language.ENGLISH) -> pd.DataFrame:
    match lang:
        case Language.ENGLISH:
            base_font_dir = EN_FONT_PATH
        case Language.JAPANESE:
            base_font_dir = JP_FONT_PATH
        case _:
            raise ValueError(f"Language not recognised: {lang}")
    
    font_data = []

    for font_name in os.listdir(base_font_dir):
        font_dir = os.path.join(base_font_dir, font_name)
        try:
            for font_path in os.listdir(font_dir):
                if font_path.endswith("ttf") or font_path.endswith("otf"):
                    font_data.append({
                        "path": os.path.join(font_dir, font_path),
                        "family": font_name,
                        "lang": lang,
                        "format": "truetype" if font_path[-3:] == "ttf" else "opentype",
                    })

        except Exception as e:
            print(f"{font_dir}: {e}")

    return pd.DataFrame(font_data)


if __name__ == "__main__":
    print(get_font_data())
