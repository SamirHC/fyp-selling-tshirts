from abc import ABC, abstractmethod
import os

import pandas as pd


class PageScraper(ABC):
    BASE_HTML_DIR = ""
    BASE_SAVE_DIR = ""

    @staticmethod
    @abstractmethod
    def scrape_html_to_dataframe(html_path: str) -> pd.DataFrame:
        pass

    @classmethod
    def scrape_directory_to_dataframe(cls) -> pd.DataFrame:
        dfs = []
        for filepath in os.listdir(cls.BASE_HTML_DIR):
            abspath = os.path.join(cls.BASE_HTML_DIR, filepath)

            df = cls.scrape_html_to_dataframe(abspath)
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True).drop_duplicates()
        df.reset_index(inplace=True)
        return df
