from abc import ABC
import os

from PIL import Image


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
