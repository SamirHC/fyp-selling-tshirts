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


class EntropySegmentation(TshirtDesignSegmentationModel):
    def extract_design_bbox(self, image: Image.Image):
        SIZE = 64
        STRIDE = 4

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
                best = x1 * resize_x, y1 * resize_y, x2 * resize_x, y2 * resize_y
            
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
    def extract_design_bbox(self, image: Image.Image):
        SIZE = 32
        CENTRALITY = 0.6
        MIN_X, MAX_X = SIZE*(1-CENTRALITY)/2, SIZE*(1+CENTRALITY)/2
        MIN_Y, MAX_Y = SIZE*(1-CENTRALITY)/2, SIZE*(1+CENTRALITY)/2
        MIN_W, MIN_H = 5, 5

        resize_x = image.width / SIZE
        resize_y = image.height / SIZE
        resized_image = image.resize((SIZE, SIZE))

        grey = np.array(resized_image.convert("L"))
        edges = cv2.Canny(grey, threshold1=50, threshold2=150)
        #Image.fromarray(edges).show()

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Find outer most design bbox
        rect = x, y, w, h = 0, 0, SIZE, SIZE
        rects = [rect]
        densities = [np.sum(edges[x:x+w, y:y+h])/(w*h)]
        for contour in contours:
            rect = x, y, w, h = cv2.boundingRect(contour)

            if w < MIN_W and h < MIN_H:
                break
            if not (MIN_X < (x+w)/2 < MAX_X):
                break
            if not (MIN_Y < (y+h)/2 < MAX_Y):
                break

            #print(rect)
            rects.append(rect)
            densities.append(np.sum(edges[x:x+w, y:y+h])/(w*h))

        if len(densities) > 1:
            delta_density = np.array([d2/d1 for d1, d2 in zip(densities, densities[1:])])
            #print(delta_density)
            x, y, w, h = rects[np.argmax(delta_density)+1]
        else:
            x, y, w, h = rects[0]
        x = int(max(0, (x-1)*resize_x))
        y = int(max(0, (y-1)*resize_y))
        w = int((w+2)*resize_x)
        h = int((h+2)*resize_y)

        return (x, y, x+w, y+h)


class ProceduralSegmentation(TshirtDesignSegmentationModel):
    def extract_design_bbox(self, image: Image.Image):
        size = 64
        resize_x = image.width / size
        resize_y = image.height / size
        res = image.resize((size, size))
        edges = image.filter(ImageFilter.FIND_EDGES).convert("1")
        edges = edges.filter(ImageFilter.MedianFilter)
        #edges.show()
        #print(np.array(edges))
        #res.show()
        arr = np.array(res)
        opaque = arr[:, :, 3] >= 200

        row_indices = np.where(np.sum(opaque, axis=1) > 32)[0]
        #print(row_indices)
        col_indices = np.where(np.sum(opaque, axis=0) > 26)[0]
        try:
            y1, y2 = np.min(row_indices) * resize_y, np.max(row_indices) * resize_y
            x1, x2 = np.min(col_indices) * resize_x, np.max(col_indices) * resize_x
            return (x1, y1, x2, y2)
        except:
            return super().extract_design_bbox(image)


class UNetSegmentation(TshirtDesignSegmentationModel):
    def __init__(self):
        model_path = os.path.join("data", "models", "TshirtPrintImageSegmentationModel1.pt")


class SegformerB3ClothesSegmentation(TshirtDesignSegmentationModel):
    def __init__(self):
        self.model = segformer_b3_clothes.SegformerB3Clothes

    def extract_design_bbox(self, image: Image.Image):
        size = 64
        resize_x = image.width / size
        resize_y = image.height / size

        mask = image.resize((size, size))
        mask.filter(ImageFilter.SHARPEN)
        mask = self.model.segment_upper_clothes(mask)
        
        arr = np.array(mask)
        row_indices = np.where(np.sum(arr, axis=1) > 32)[0]
        col_indices = np.where(np.sum(arr, axis=0) > 26)[0]
        try:
            y1, y2 = np.min(row_indices) * resize_y, np.max(row_indices) * resize_y
            x1, x2 = np.min(col_indices) * resize_x, np.max(col_indices) * resize_x
            assert x1 < x2 and y1 < y2
            return (x1, y1, x2, y2)
        except:
            return super().extract_design_bbox(image)


class LLMSegmentation(TshirtDesignSegmentationModel):
    def __init__(self):
        self.model = DeepSeekLLM()
    def extract_design_bbox(self, image: Image.Image):
        SIZE = 16

        resize_x = image.width / SIZE
        resize_y = image.height / SIZE
        resized_image = image.resize((SIZE, SIZE))

        grey = np.array(resized_image.convert("L"))
        edges = cv2.Canny(grey, threshold1=50, threshold2=150)
        edge_string = "\n".join("".join(str(int(item == 255)) for item in row) for row in edges)
        print(edge_string)
        
        completion = self.model.client.chat.completions.create(
            model="deepseek/deepseek-r1:free",
            messages=[
                {"role": "system",
                 "content": "ONLY RESPOND WITH A PYTHON TUPLE IN THE FORMAT (x, y, width, height). NO EXPLANATIONS OR EXTRA TEXT."},
                {"role": "user",
                "content": f"Given the Tshirt edge binary mask, return the bounding box of the Tshirt, excluding the sleeves:\n{edge_string}."}
            ],
        )
        print(completion)
        result = str(completion.choices[0].message.content)
        print(result)
        success = True
        try:
            stripped = result.strip("() ")
            if not stripped:
                success = False            
            result = tuple(int(num.strip()) for num in stripped.split(","))
            success = len(result) == 4
        except:
            success = False

        if success:
            x, y, w, h = result
            x *= resize_x
            y *= resize_y
            w *= resize_x
            h *= resize_y

            return (x, y, w, h)
        else:
            return super().extract_design_bbox(image)


if __name__ == "__main__":
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
