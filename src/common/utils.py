import pickle

import base64
import io
import pandas as pd
from PIL import Image
import requests


def save_data(df: pd.DataFrame, save_path):
    """
    Saves the dataframe to a pickle file.
    
    :param df: the dataframe to save
    :param save_path: path to save to
    """

    with open(save_path, "wb") as target:
        pickle.dump(df, target)

    print(f"Saved data to {save_path}")


def load_data(path) -> pd.DataFrame:
    """
    Loads the pickle file into a dataframe.

    :param path: path to load from

    :return df: the dataframe contents of the pickle file
    """

    with open(path, "rb") as target:
        df = pickle.load(target)

    print(f"Loaded data from {path}\n")
    return df


def get_image_from_url(url):
    """
    Loads an image given a url, or None if it fails.

    :param url: URL to load the image from

    :return image: Image object
    """

    response = requests.get(url, stream=True)

    if response.status_code == 200:
        return Image.open(response.raw)
    else:
        print(f"Failed to retrieve the image. Status code: {response.status_code}")


def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
    