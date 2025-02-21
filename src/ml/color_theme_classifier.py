import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from src.common import utils
from src.data_collection import palettes


class ColorThemeClassifier:
    color_df = palettes.get_palette_data()

    @staticmethod
    def extract_colors(image, seed=23920):
        pixels = np.array(image).reshape(-1, 3)

        num_colors = 4
        kmeans = KMeans(n_clusters=num_colors, n_init=10, random_state=seed)
        kmeans.fit(pixels)

        colors = kmeans.cluster_centers_.astype(np.uint8)
        return colors

    @staticmethod
    def find_closest_palette(colors):
        weights = np.ones((4, 1), dtype=np.float32)

        colors = np.array(colors, dtype=np.float32)
        colors = np.hstack((weights, colors))

        min_dist = float('inf')
        row_idx = -1
        for i in range(len(ColorThemeClassifier.color_df)):
            row = ColorThemeClassifier.color_df.iloc[i]
            row_colors = np.array([palettes.hex_to_rgb(hex) for hex in row["colors"]], dtype=np.float32)
            row_colors = np.hstack((weights, row_colors))

            dist, _, _ = cv2.EMD(row_colors, colors, cv2.DIST_L2)
            if dist < min_dist:
                min_dist = dist
                row_idx = i
                nearest_colors = row_colors

        nearest_colors = np.array(nearest_colors[:, 1:], dtype=np.uint8)
        return row_idx, nearest_colors, min_dist

    @staticmethod
    def get_palette_data(image) -> pd.Series:
        colors_rgb = ColorThemeClassifier.extract_colors(image)
        row_idx, nearest_colors, dist = ColorThemeClassifier.find_closest_palette(colors_rgb)
        
        row = ColorThemeClassifier.color_df.iloc[row_idx]
        colors_rgb = [tuple(color) for color in colors_rgb]
        nearest_colors = [tuple(color) for color in nearest_colors]

        return pd.Series({
            "row_idx": row_idx,
            "colors": colors_rgb,
            "nearest_colors": nearest_colors,
            "dist": dist,
            "color_tags": row["color_tags"],
            "other_tags": row["other_tags"],
        })

    @staticmethod
    def show_data(image):
        palette_data = ColorThemeClassifier.get_palette_data(image)
        print(palette_data)

        colors = palette_data["colors"]
        colors = np.array(sorted(colors), dtype=np.uint8).reshape(1, -1, 3)

        nearest_colors = palette_data["nearest_colors"]
        nearest_colors = np.array(sorted(nearest_colors), dtype=np.uint8).reshape(1, -1, 3)

        _, axs = plt.subplots(1, 3)

        axs[0].imshow(colors)
        axs[0].set_title("Extracted Colors")
        axs[1].imshow(nearest_colors)
        axs[1].set_title("Nearest Colors")
        axs[2].imshow(image)

        plt.show()

if __name__ == "__main__":
    df = utils.load_data("data/dataframes/seller_hub_data/ebay_data.pickle")
    image = utils.get_image_from_url(df["img_url"][17])

    ColorThemeClassifier.show_data(image)
