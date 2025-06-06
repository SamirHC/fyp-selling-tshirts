import itertools

from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from scipy import stats

from src.ml.tshirt_design_segmentation.segmentation import TshirtDesignSegmentationModel


class EntropySegmentation(TshirtDesignSegmentationModel):
    def __init__(self, grid_size, num_clusters):
        assert isinstance(grid_size, int)
        # Hyperparams
        self.grid_size = grid_size
        self.num_clusters = num_clusters
    
    def extract_design_bbox(self, image: Image.Image):
        DX = image.width // self.grid_size
        DY = image.height // self.grid_size

        pixels = np.array(image).reshape(-1, 3)
        kmeans = KMeans(self.num_clusters)
        kmeans.fit(pixels)
        clustered = kmeans.labels_.reshape(image.size)

        grid_counts = np.zeros((self.grid_size, self.grid_size, self.num_clusters))
        for i in range(self.grid_size):
            i1, i2 = i*DX, (i+1)*DX
            for j in range(self.grid_size):
                j1, j2 = j*DY, (j+1)*DY
                grid_counts[i, j, :] = np.bincount(
                    clustered[i1:i2, j1:j2].ravel(),
                    minlength=self.num_clusters
                )

        optimum_impurity = float('inf')
        best = 0, 0, DX * self.grid_size, DY * self.grid_size
        regions = itertools.product(
            range(0, self.grid_size//2),
            range(0, self.grid_size//2),
            range(self.grid_size//2, self.grid_size+1),
            range(self.grid_size//2, self.grid_size+1),
        )
        for region in regions:
            x1, y1, x2, y2 = region
            inner = np.sum(grid_counts[x1:x2, y1:y2], axis=(0, 1))
            outer = (
                np.sum(grid_counts[:, :y1], axis=(0, 1)) + 
                np.sum(grid_counts[:x1, y1:y2], axis=(0, 1)) +
                np.sum(grid_counts[x2:, y1:y2], axis=(0, 1)) +
                np.sum(grid_counts[:, y2:], axis=(0, 1))
            )
            val = np.sum(inner)*stats.entropy(inner, base=2) + np.sum(outer)*stats.entropy(outer, base=2)
            if val < optimum_impurity:
                optimum_impurity = val
                best = region

        x1, y1, x2, y2 = best
        x1 *= DX
        y1 *= DY
        x2 *= DX
        y2 *= DY
        return (x1,y1,x2,y2)


def main():
    from src.common import utils
    import os

    seg_model = EntropySegmentation(16, 5)
    
    image_df_path = os.path.join("data", "dataframes", "seller_hub_data", "ebay_data.pickle")
    tshirt_df = utils.load_data(image_df_path)

    for i in range(len(tshirt_df)):
        url = tshirt_df.iloc[i]["img_url"]
        image = utils.get_image_from_url(url).convert("RGB")
        seg_image = seg_model.extract_design(image)
        seg_image.show()
        input()


if __name__ == "__main__":
    main()
