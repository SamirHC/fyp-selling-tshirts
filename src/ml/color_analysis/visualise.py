from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import skimage

from src.common import utils
from src.data_collection import palettes
from src.ml.color_analysis.color_theme_classifier import RGBColorThemeClassifier, CIELabColorThemeClassifier


def plot_rgb_pixels(image: Image.Image, n=2000) -> plt.Figure:
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


def plot_cielab_pixels(image: Image.Image, n=2000) -> plt.Figure:
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
    N = 11
    # RGB
    rgb_kmeans_colors = RGBColorThemeClassifier.extract_colors(image)
    rgb_distances_df = RGBColorThemeClassifier.compute_distances(rgb_kmeans_colors)
    rgb_min = rgb_distances_df.nsmallest(N, "dist")["nearest_colors"]
    rgb_min = rgb_min.apply(lambda x: x.astype(np.uint8).reshape(1, -1, 3))

    # CIELab
    cielab_kmeans_colors = CIELabColorThemeClassifier.extract_colors(image)
    cielab_min = CIELabColorThemeClassifier.get_k_nearest_palettes(cielab_kmeans_colors, k=N)

    print(rgb_min)
    print(cielab_min)

    nrows, ncols = 2, N + 2
    fig, axs = plt.subplots(nrows, ncols)

    axs[0][0].set_title("Extracted Colors (RGB)")
    axs[0][0].imshow(rgb_kmeans_colors.reshape(-1, 1, 3))

    axs[1][0].set_title("Extracted Colors (CIELab)")
    axs[1][0].imshow(cielab_kmeans_colors.reshape(-1, 1, 3))

    axs[0][2 + N//2].set_title("Nearest Colors (RGB)")
    axs[1][2 + N//2].set_title("Nearest Colors (CIELab)")
    for i in range(N):
        axs[0][i+2].imshow(rgb_min.iloc[i].reshape(-1, 1, 3))
        axs[1][i+2].imshow(cielab_min.iloc[i].reshape(-1, 1, 3))

    for i in range(nrows):
        for j in range(ncols):
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
            if j == 1:
                axs[i][j].set_axis_off()

    return fig


if __name__ == "__main__":
    image = utils.get_image_from_url("https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/456433/sub/goods_456433_sub20_3x4.jpg?width=600")    
    image = image.crop((0, 100, 600, 700))

    plot_rgb_pixels(image, n=5000)
    plot_cielab_pixels(image, n=5000)
    plt.show()

    show_data(image, RGBColorThemeClassifier)
    show_data(image, CIELabColorThemeClassifier)
    plt.show()

    show_data_2(image)
    plt.show()
