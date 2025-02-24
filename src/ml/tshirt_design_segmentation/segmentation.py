from abc import ABC
import os

from PIL import Image

from src.ml.tshirt_design_segmentation import segformer_b3_clothes


class TshirtDesignSegmentationModel(ABC):
    def extract_design(image):
        return image


class ProceduralSegmentation(TshirtDesignSegmentationModel):
    def extract_design(self, image: Image.Image):
        # TODO
        return image


class UNetSegmentation(TshirtDesignSegmentationModel):
    def __init__(self):
        model_path = os.path.join("data", "models", "TshirtPrintImageSegmentationModel1.pt")

    def extract_design(self, image: Image.Image):
        # TODO
        return image


class SegformerB3ClothesSegmentation(TshirtDesignSegmentationModel):
    def __init__(self):
        self.model = segformer_b3_clothes.SegformerB3Clothes

    def extract_design(self, image: Image.Image):
        return self.model.segment_upper_clothes(image)
