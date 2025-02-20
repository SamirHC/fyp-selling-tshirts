import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from src.common import utils
from src.data_collection import palettes


color_df = palettes.get_palette_data()

def extract_colors(image):
    seed = 23920

    pixels = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR).reshape(-1, 3)

    num_colors = 4
    kmeans = KMeans(n_clusters=num_colors, n_init=10, random_state=seed)
    kmeans.fit(pixels)

    colors_bgr = kmeans.cluster_centers_.astype("uint8")
    print(f"BGR: {colors_bgr}")
    colors_rgb = cv2.cvtColor(colors_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
    print(f"RGB: {colors_rgb}")
    print("Extracted Colors (RGB):", colors_rgb)
    return colors_rgb


def find_closest_palette(colors_rgb):
    weights = np.ones((4, 1), dtype=np.float32)

    colors_rgb = np.array(colors_rgb, dtype=np.float32)
    colors_rgb = np.hstack((weights, colors_rgb))

    min_dist = float('inf')
    row_idx = -1
    for i in range(len(color_df)):
        row = color_df.iloc[i]
        row_colors = np.array([palettes.hex_to_rgb(hex) for hex in row["colors"]], dtype=np.float32)
        row_colors = np.hstack((weights, row_colors))

        dist, _, _ = cv2.EMD(row_colors, colors_rgb, cv2.DIST_L2)
        if dist < min_dist:
            min_dist = dist
            row_idx = i
            nearest_colors = row_colors

    nearest_colors = np.array(nearest_colors[:, 1:], dtype=np.uint8).reshape(1, -1, 3)
    return row_idx, nearest_colors, min_dist


if __name__ == "__main__":
    df = utils.load_data("data/dataframes/seller_hub_data/ebay_data.pickle")
    image = utils.get_image_from_url(df["img_url"][17])

    colors_rgb = extract_colors(image)
    row_idx, nearest_colors, dist = find_closest_palette(colors_rgb)

    fig, axs = plt.subplots(2, 1)
    colors_rgb = np.array(colors_rgb, dtype=np.uint8).reshape(1, -1, 3)
    axs[0].imshow(colors_rgb)
    axs[1].imshow(nearest_colors)

    image.show()
    plt.show()

    print(f"Row: {row_idx}, Dist: {dist}\n{color_df.iloc[row_idx][["color_tags","other_tags"]]}")
