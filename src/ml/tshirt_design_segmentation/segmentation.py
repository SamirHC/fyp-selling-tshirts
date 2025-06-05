from abc import ABC
import itertools
import os

import cv2
from PIL import Image, ImageFilter
import numpy as np

from src.ml.tshirt_design_segmentation import segformer_b3_clothes
from src.ml.genai.text_gen import DeepSeekLLM


class TshirtDesignSegmentationModel(ABC):
    def extract_design(self, image: Image.Image) -> Image.Image:
        return image.crop(self.extract_design_bbox(image))

    def extract_design_bbox(self, image: Image.Image) -> tuple:
        return (0, 0, image.width, image.height)


class NoSegmentation(TshirtDesignSegmentationModel):
    pass


class FixedSegmentation(TshirtDesignSegmentationModel):
    def __init__(self, x=1/4, y=1/4, w=1/2, h=1/2):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    
    def extract_design_bbox(self, image):
        x = image.width * self.x
        y = image.height * self.y
        w = x + image.width * self.w
        h = y + image.height * self.h
        return (x, y, x+w, y+h)


class EntropySegmentation(TshirtDesignSegmentationModel):
    def __init__(self, size=64, stride=4):
        assert isinstance(size, int)
        assert isinstance(stride, int)
        # Hyperparams
        self.size = size
        self.stride = stride
    
    def extract_design_bbox(self, image: Image.Image):
        SIZE = self.size
        STRIDE = self.stride

        resize_x = image.width / SIZE
        resize_y = image.height / SIZE
        resized_image = image.resize((SIZE, SIZE))

        grey = np.array(resized_image.convert("L"))

        best = 0, SIZE, 0, SIZE
        optimum = float('inf')
        regions = itertools.product(
            range(0, SIZE//2, STRIDE),
            range(0, SIZE//2, STRIDE),
            range(SIZE//2, SIZE, STRIDE),
            range(SIZE//2, SIZE, STRIDE)
        )
        for x1, y1, x2, y2 in regions:
            inner = grey[x1:x2, y1:y2].flatten()
            outer = np.concatenate((
                grey[:, :y1].flatten(),
                grey[:x1, y1:y2].flatten(),
                grey[x2:, y1:y2].flatten(),
                grey[:, y2:].flatten()
            ))

            val = len(inner)*self.entropy(inner) + len(outer)*self.entropy(outer)
            if val < optimum:
                optimum = val
                best = (
                    int(x1 * resize_x),
                    int(y1 * resize_y),
                    int(x2 * resize_x),
                    int(y2 * resize_y),
                )
            
            #print(f"Best: {best}, Opt: {optimum}, {x1,x2,y1,y2}")
        
        return best
    
    def entropy(self, data):
        EPS = 1e-10
        hist, bins = np.histogram(data, bins=32, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        delta_z = bin_centers[1] - bin_centers[0]
        entropy = -np.sum(hist * np.log2(hist + EPS) * delta_z)
        return entropy


class ContourSegmentation(TshirtDesignSegmentationModel):
    def __init__(self, size=32, centrality=0.8, min_w=1/8, min_h=1/8):
        # Hyperparams
        self.size = size
        self.centrality = centrality
        self.min_w = min_w
        self.min_h = min_h
    
    def extract_design_bbox(self, image: Image.Image):
        SIZE = self.size
        CENTRALITY = self.centrality
        MIN_X, MAX_X = SIZE*(1-CENTRALITY)/2, SIZE*(1+CENTRALITY)/2
        MIN_Y, MAX_Y = SIZE*(1-CENTRALITY)/2, SIZE*(1+CENTRALITY)/2
        MIN_W, MIN_H = int(SIZE*self.min_w), int(SIZE*self.min_h)

        def _valid_contour(contour) -> bool:
            x, y, w, h = contour
            return all((
                MIN_W < w, 
                MIN_H < h,
                MIN_X < x < x + w < MAX_X,
                MIN_Y < y < y + h < MAX_Y,
            ))

        resize_x = image.width / SIZE
        resize_y = image.height / SIZE
        resized_image = image.resize((SIZE, SIZE))

        grey = np.array(resized_image.convert("L"))
        edges = cv2.Canny(grey, threshold1=50, threshold2=150)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contours = map(cv2.boundingRect, contours)
        contours = filter(_valid_contour, contours)

        x, y, w, h = next(contours, (0, 0, SIZE, SIZE))
        
        x = int(max(0, x*resize_x))
        y = int(max(0, y*resize_y))
        w = int(w*resize_x)
        h = int(h*resize_y)
        return (x, y, x+w, y+h)


class UNetSegmentation(TshirtDesignSegmentationModel):
    def __init__(self):
        model_path = os.path.join("data", "models", "TshirtPrintImageSegmentationModel1.pt")


class SegformerB3ClothesSegmentation(TshirtDesignSegmentationModel):
    def __init__(self, width_cut=5, top_cut=8, bottom_cut=10):
        self.model = segformer_b3_clothes.SegformerB3Clothes
        # Hyperparameters
        self.width_cut = width_cut
        self.top_cut = top_cut
        self.bottom_cut = bottom_cut

    def extract_design_bbox(self, image: Image.Image):
        mask = self.model.segment_upper_clothes(image)
        arr = np.array(mask)
        rows, cols = np.where(arr)
        if len(rows) > 0 and len(cols) > 0:
            y1, y2 = rows.min(), rows.max()
            x1, x2 = cols.min(), cols.max()
            w = x2 - x1
            x1 += w // 5
            x2 -= w // 5
            h = y2 - y1
            y1 += h // 8
            y2 -= h // 10
            assert x1 < x2 and y1 < y2
            return (int(x1), int(y1), int(x2), int(y2))
        else:
            return super().extract_design_bbox(image)


def main():
    from src.common import utils

    seg_model = ContourSegmentation()
    
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