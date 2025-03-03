import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import skimage
import matplotlib.pyplot as plt
from PIL import Image

from src.common import utils
from src.data_collection import palettes


class ColorThemeClassifier:
    color_df = palettes.get_palette_data()

    @staticmethod
    def extract_colors(image: Image.Image, seed=23920):
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
    def find_closest_palette_cielab(colors):
        weights = np.ones((4, 1), dtype=np.float32)

        colors = np.array(colors, dtype=np.float32).reshape(1, 4, 3) / 255
        colors = skimage.color.rgb2lab(colors)
        colors = np.hstack((weights, colors.reshape(-1, 3)))
        print(colors)

        min_dist = float('inf')
        row_idx = -1
        for i in range(len(ColorThemeClassifier.color_df)):
            row = ColorThemeClassifier.color_df.iloc[i]
            row_colors = np.array([palettes.hex_to_rgb(hex) for hex in row["colors"]], dtype=np.float32).reshape(1, 4, 3) / 255
            row_colors = skimage.color.rgb2lab(row_colors)
            row_colors = np.hstack((weights, row_colors.reshape(-1, 3)))

            dist, _, flow = cv2.EMD(row_colors, colors, cv2.DIST_L2)
            if dist < min_dist:
                min_dist = dist
                row_idx = i
                nearest_colors = flow @ row_colors[:, 1:]

        nearest_colors = nearest_colors.reshape(1, 4, 3)
        nearest_colors = skimage.color.lab2rgb(nearest_colors) * 255
        nearest_colors = nearest_colors.reshape(-1, 3).astype(np.uint8)

        return row_idx, nearest_colors, min_dist


    @staticmethod
    def get_palette_data(image: Image.Image) -> pd.Series:
        colors_rgb = ColorThemeClassifier.extract_colors(image)
        row_idx, nearest_colors, dist = ColorThemeClassifier.find_closest_palette_cielab(colors_rgb)
        
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
    def show_data(image: Image.Image):
        palette_data = ColorThemeClassifier.get_palette_data(image)
        print(palette_data)

        colors = palette_data["colors"]
        colors = np.array(colors, dtype=np.uint8).reshape(1, -1, 3)

        nearest_colors = palette_data["nearest_colors"]
        nearest_colors = np.array(nearest_colors, dtype=np.uint8).reshape(1, -1, 3)

        _, axs = plt.subplots(1, 3)

        axs[0].imshow(colors)
        axs[0].set_title("Extracted Colors")
        axs[1].imshow(nearest_colors)
        axs[1].set_title("Nearest Colors")
        axs[2].imshow(image)

        plt.show()
    
    @staticmethod
    def plot_pixel_colors(image: Image.Image, n=2000) -> plt.Figure:
        colors = np.array(image).reshape(-1, 3)
        
        # Uniform sampling of the colours to reduce number of pixels plotted
        indices = np.arange(len(colors))
        np.random.shuffle(indices)
        indices = indices[:n]

        colors = colors[indices, :]

        r = colors[:, 0]
        g = colors[:, 1]
        b = colors[:, 2]
        
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.set_xlabel("Red")
        ax.set_ylabel("Blue")
        ax.set_zlabel("Green")

        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
        ax.set_zlim(0, 255)

        ax.scatter(r, g, b, c=colors/255)

        return fig


if __name__ == "__main__":
    image = utils.get_image_from_url("https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/456433/sub/goods_456433_sub20_3x4.jpg?width=600")    
    image = image.crop((0, 100, 600, 700))

    ColorThemeClassifier.plot_pixel_colors(image, n=5000)

    plt.show()

    ColorThemeClassifier.show_data(image)
