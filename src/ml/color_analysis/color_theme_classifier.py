import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import optimize
import skimage
from sklearn.cluster import KMeans

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
        return RGBColorThemeClassifier.palette_distance_hungarian(p, q)

    @staticmethod
    def palette_distance_EMD(p: np.ndarray, q: np.ndarray) -> tuple[float, np.ndarray]:
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
    def palette_distance_hungarian(p: np.ndarray, q: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Calculates the distance between two 4-color palettes.
        It also returns the optimal pairings based on the computed flow matrix.

        :param p: A (4, 3) numpy array representing the first color palette.
        :param q: A (4, 3) numpy array representing the second color palette.

        :return: A tuple containing:

            - **dist (float)**: 
                The computed distance between `p` and `q`.
            - **nearest_colors (np.ndarray)**:
                A (4, 3) numpy array representing permuted nearest colors in `q`
                to `p`.
        """

        p = p.astype(np.float32)
        q = q.astype(np.float32)

        cost_matrix = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                cost_matrix[i, j] = np.linalg.norm(p[i] - q[j])

        row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
        flow = np.zeros((4, 4))
        flow[row_ind, col_ind] = 1

        nearest_colors = flow @ q
        dist = np.sum(cost_matrix[row_ind, col_ind])        
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
            lambda row_colors: pd.Series(
                RGBColorThemeClassifier.palette_distance(
                    colors, 
                    palettes.hex_palette_to_rgb_array(row_colors)
                )
            )
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
        return CIELabColorThemeClassifier.palette_distance_hungarian(p, q)

    @staticmethod
    def palette_distance_hungarian(p: np.ndarray, q: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Calculates the distance between two 4-color palettes.
        It also returns the optimal pairings based on the computed flow matrix.

        :param p: A (4, 3) numpy array representing the first color palette.
        :param q: A (4, 3) numpy array representing the second color palette.

        :return: A tuple containing:

            - **dist (float)**: 
                The computed distance between `p` and `q`.
            - **nearest_colors (np.ndarray)**:
                A (4, 3) numpy array representing permuted nearest colors in `q`
                to `p`.
        """

        cost_matrix = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                cost_matrix[i, j] = np.linalg.norm(p[i, :] - q[j, :])

        row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
        flow = np.zeros((4, 4))
        flow[row_ind, col_ind] = 1

        nearest_colors = flow @ q
        dist = np.sum(cost_matrix[row_ind, col_ind])
        return dist, nearest_colors

    @staticmethod
    def palette_distance_EMD(p: np.ndarray, q: np.ndarray) -> tuple[float, np.ndarray]:
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
