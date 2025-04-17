import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import skimage
from sklearn.cluster import KMeans

from src.common import utils
from src.data_collection import palettes


color_df = palettes.get_palette_data()


class RGBColorThemeClassifier:
    @staticmethod
    def extract_colors(image: Image.Image, seed=23920) -> np.ndarray:
        """
        Extract the four dominant colors from an image using K-Means clustering
        in RGB color space.

        :param image: A PIL Image object from which to extract colors.
        :param seed: Random seed for K-Means clustering to ensure reproducibility.

        
        :return colors: A numpy array of shape (4, 3) containing the (0-255) RGB values of the 
                        four dominant colors found in the image.
        """

        pixels = np.array(image).reshape(-1, 3)

        num_colors = 4
        kmeans = KMeans(n_clusters=num_colors, n_init=10, random_state=seed)
        kmeans.fit(pixels)

        colors = kmeans.cluster_centers_.astype(np.uint8)
        return colors

    @staticmethod
    def palette_distance(p: np.ndarray, q: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Calculates the Earth Mover's Distance between two 4-color palettes using
        the L2-norm (Euclidean Distance).
        It also returns the optimal pairings based on the computed flow matrix.

        :param p: A (4, 3) numpy array representing the first color palette.
        :param q: A (4, 3) numpy array representing the second color palette.

        :return: A tuple containing:

            - **dist (float)**: 
                The computed Earth Mover's Distance (EMD) between `p` and `q`.
            - **nearest_colors (np.ndarray)**:
                A (4, 3) numpy array representing permuted nearest colors in `q`
                to `p`.
        """

        weights = np.ones((4, 1))
        p = np.hstack((weights, p), dtype=np.float32)
        q = np.hstack((weights, q), dtype=np.float32)

        dist, _, flow = cv2.EMD(p, q, cv2.DIST_L2)
        nearest_colors = flow @ q[:, 1:]
        return dist, nearest_colors

    @staticmethod
    def find_closest_palette(colors):
        distances_df = RGBColorThemeClassifier.compute_distances(colors)
        min_row = distances_df.loc[distances_df["dist"].idxmin()]

        min_dist = min_row["dist"]
        row_idx = min_row.name
        nearest_colors = min_row["nearest_colors"]

        return row_idx, nearest_colors, min_dist

    @staticmethod
    def compute_distances(colors):
        df = color_df.copy()
        df[["dist", "nearest_colors"]] = df["colors"].apply(
            lambda row_colors: pd.Series(RGBColorThemeClassifier.palette_distance(colors, palettes.hex_palette_to_rgb_array(row_colors)))
        )
        return df

    @staticmethod
    def get_palette_data(image: Image.Image) -> pd.Series:
        colors = RGBColorThemeClassifier.extract_colors(image)
        row_idx, nearest_colors, dist = RGBColorThemeClassifier.find_closest_palette(colors)
        nearest_colors = nearest_colors.astype(np.uint8)

        row = color_df.iloc[row_idx]
        colors = [tuple(color) for color in colors]
        nearest_colors = [tuple(color) for color in nearest_colors]

        return pd.Series({
            "row_idx": row_idx,
            "colors": colors,
            "nearest_colors": nearest_colors,
            "dist": dist,
            "color_tags": row["color_tags"],
            "other_tags": row["other_tags"],
        })


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
        ax.set_ylabel("Green")
        ax.set_zlabel("Blue")

        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
        ax.set_zlim(0, 255)

        ax.scatter(r, g, b, c=colors/255)

        return fig


class CIELabColorThemeClassifier:
    @staticmethod
    def extract_colors(image: Image.Image, seed=23920) -> np.ndarray:
        """
        Extract the four dominant colors from an image using K-Means clustering
        in CIELab color space.

        :param image: A PIL Image object from which to extract colors.
        :param seed: Random seed for K-Means clustering to ensure reproducibility.

        :return colors: A numpy array of shape (4, 3) containing the (0-255) RGB
            values of the four dominant colors found in the image.
        """

        pixels = np.array(image).reshape(-1, 3)
        cielab_pixels = skimage.color.rgb2lab(pixels / 255)

        num_colors = 4
        kmeans = KMeans(n_clusters=num_colors, n_init=10, random_state=seed)
        kmeans.fit(cielab_pixels)

        cielab_colors = kmeans.cluster_centers_
        colors = (skimage.color.lab2rgb(cielab_colors) * 255).astype(np.uint8)
        return colors

    @staticmethod
    def palette_distance(p: np.ndarray, q: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Calculates the Earth Mover's Distance between two 4-color palettes using
        the L2-norm (Euclidean Distance). 
        It also returns the optimal pairings based on the computed flow matrix.

        :param p: A (4, 3) numpy array representing the first color palette.
        :param q: A (4, 3) numpy array representing the second color palette.

        :return: A tuple containing:

            - **dist (float)**:
                The computed Earth Mover's Distance (EMD) between `p` and `q`.
            - **nearest_colors (np.ndarray)**:
                A (4, 3) numpy array representing permuted nearest colors in `q`
                to `p`.
        """

        weights = np.ones((4, 1))
        p = np.hstack((weights, p), dtype=np.float32)
        q = np.hstack((weights, q), dtype=np.float32)

        dist, _, flow = cv2.EMD(p, q, cv2.DIST_L2)
        nearest_colors = flow @ q[:, 1:]
        return dist, nearest_colors

    @staticmethod
    def find_closest_palette(colors):
        distances_df = CIELabColorThemeClassifier.compute_distances(colors)
        min_row = distances_df.loc[distances_df["dist"].idxmin()]

        min_dist = min_row["dist"]
        row_idx = min_row.name
        nearest_colors = min_row["nearest_colors"]

        return row_idx, nearest_colors, min_dist

    @staticmethod
    def compute_distances(colors):
        df = color_df.copy()
        df[["dist", "nearest_colors"]] = df["colors"].apply(
            lambda row_colors: pd.Series(
                CIELabColorThemeClassifier.palette_distance(
                    colors,
                    palettes.hex_palette_to_cielab_array(row_colors)
                )
            )
        )
        return df

    @staticmethod
    def get_palette_data(image: Image.Image) -> pd.Series:
        colors = CIELabColorThemeClassifier.extract_colors(image)
        row_idx, nearest_colors, dist = (
            CIELabColorThemeClassifier.find_closest_palette(
                palettes.rgb_array_palette_to_cielab_array(colors)
            )
        )
        nearest_colors = (skimage.color.lab2rgb(nearest_colors) * 255).astype(np.uint8)

        row = color_df.iloc[row_idx]
        colors = [tuple(color) for color in colors]
        nearest_colors = [tuple(color) for color in nearest_colors]

        return pd.Series({
            "row_idx": row_idx,
            "colors": colors,
            "nearest_colors": nearest_colors,
            "dist": dist,
            "color_tags": row["color_tags"],
            "other_tags": row["other_tags"],
        })

    @staticmethod
    def plot_pixel_colors(image: Image.Image, n=2000) -> plt.Figure:
        colors = np.array(image).reshape(-1, 3) / 255
        cielab_colors = skimage.color.rgb2lab(colors)

        # Uniform sampling of the colours to reduce number of pixels plotted
        indices = np.arange(len(colors))
        np.random.shuffle(indices)
        indices = indices[:n]

        colors = colors[indices, :]
        cielab_colors = cielab_colors[indices, :]

        l = cielab_colors[:, 0]
        a = cielab_colors[:, 1]
        b = cielab_colors[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.set_xlabel("L*")
        ax.set_ylabel("a*")
        ax.set_zlabel("b*")

        ax.set_xlim(0, 100)
        ax.set_ylim(-128, 127)
        ax.set_zlim(-128, 127)

        ax.scatter(l, a, b, c=colors)

        return fig


def show_data(image: Image.Image, classifier) -> plt.Figure:
    palette_data = classifier.get_palette_data(image)
    print(palette_data)

    colors = np.array(palette_data["colors"]).reshape(1, -1, 3)
    nearest_colors = np.array(palette_data["nearest_colors"]).reshape(1, -1, 3)

    fig, axs = plt.subplots(1, 3)

    axs[0].imshow(colors)
    axs[0].set_title("Extracted Colors")
    axs[1].imshow(nearest_colors)
    axs[1].set_title("Nearest Colors")
    axs[2].imshow(image)

    for i in range(3):
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    return fig


def show_data_2(image: Image.Image) -> plt.Figure:
    N = 7
    # RGB
    colors = RGBColorThemeClassifier.extract_colors(image)
    rgb_distances_df = RGBColorThemeClassifier.compute_distances(colors)
    rgb_min = rgb_distances_df.nsmallest(N, "dist")["nearest_colors"]
    rgb_min = rgb_min.apply(lambda x: x.astype(np.uint8).reshape(1, -1, 3))

    # CIELab
    colors_ = CIELabColorThemeClassifier.extract_colors(image)
    colors_ = palettes.rgb_array_palette_to_cielab_array(colors_)
    cielab_distances_df = CIELabColorThemeClassifier.compute_distances(colors_)
    cielab_min = cielab_distances_df.nsmallest(N, "dist")["nearest_colors"]
    cielab_min = cielab_min.apply(lambda x: (skimage.color.lab2rgb(x) * 255).astype(np.uint8).reshape(1, -1, 3))

    print(rgb_min)
    print(cielab_min)

    nrows, ncols = N, 3
    fig, axs = plt.subplots(nrows, ncols)

    axs[N//2][0].set_title("Extracted Colors")
    axs[N//2][0].imshow(colors.reshape(1, -1, 3))

    axs[0][1].set_title("Nearest Colors (RGB)")
    axs[0][2].set_title("Nearest Colors (CIELab)")
    for i in range(N):
        axs[i][1].imshow(rgb_min.iloc[i])
        axs[i][2].imshow(cielab_min.iloc[i])

    for i in range(nrows):
        for j in range(ncols):
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
            if i != N//2 and j == 0:
                axs[i][j].set_axis_off()

    return fig


if __name__ == "__main__":
    image = utils.get_image_from_url("https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/456433/sub/goods_456433_sub20_3x4.jpg?width=600")    
    image = image.crop((0, 100, 600, 700))

    RGBColorThemeClassifier.plot_pixel_colors(image, n=5000)
    CIELabColorThemeClassifier.plot_pixel_colors(image, n=5000)
    plt.show()

    show_data(image, RGBColorThemeClassifier)
    show_data(image, CIELabColorThemeClassifier)
    plt.show()

    show_data_2(image)
    plt.show()
