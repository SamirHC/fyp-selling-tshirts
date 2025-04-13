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


def get_image_from_url(url: str) -> Image.Image:
    """
    Retrieves an image from a given URL and returns it as a Pillow Image object.
    If the image cannot be loaded, it returns an empty image with a size of (0, 0).

    :param url: The URL from which to load the image.
    :type url: str
    :returns: A Pillow Image object containing the image retrieved from the URL.
    :rtype: PIL.Image.Image
    :raises ValueError: If the URL is invalid or the image cannot be retrieved.
    """

    response = requests.get(url, stream=True)

    if response.status_code == 200:
        return Image.open(response.raw)
    else:
        raise ValueError(f"Failed to retrieve the image. Status code: {response.status_code}")


def image_to_base64(image: Image.Image) -> str:
    """
    Converts an image (PIL Image object) into a base64-encoded string.

    :param image: A Pillow Image object to be converted.
    :type image: PIL.Image.Image
    :returns: A base64-encoded string representation of the image in PNG format.
    :rtype: str
    """

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
