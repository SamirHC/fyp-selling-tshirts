import pickle

import pandas as pd


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
