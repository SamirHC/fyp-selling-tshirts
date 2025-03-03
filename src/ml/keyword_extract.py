from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class TFIDF:
    def __init__(self, corpus):
        self.corpus = corpus

        vectoriser = TfidfVectorizer()
        self.result = vectoriser.fit_transform(corpus)
        self.feature_names = vectoriser.get_feature_names_out()

    def get_top_keywords(self, corpus_index, n=5):
        tfidf_scores = self.result[corpus_index].toarray().flatten()
        top_indices = np.argsort(tfidf_scores)[::-1][:n]
        top_keywords = [self.feature_names[i] for i in top_indices]

        return top_keywords


if __name__ == "__main__":
    import os
    from src.common import utils

    image_df_path = os.path.join("data", "dataframes", "etsy_listing_data", "EtsyPageScraper 2025-01-08 08:53:59.pickle")
    tshirt_df = utils.load_data(image_df_path)

    tfidf = TFIDF(tshirt_df["title"])
    for i in range(len(tshirt_df)):
        print(tshirt_df.iloc[i]["title"])
        print(tfidf.get_top_keywords(i))
        print()
