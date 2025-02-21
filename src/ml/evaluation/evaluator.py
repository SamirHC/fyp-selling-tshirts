from abc import ABC

from PIL import Image


class Score:
    def __init__(self, value: float):
        self.value = value


class Evaluator(ABC):
    def evaluate(self, image: Image.Image) -> Score:
        return Score(0.0)


class DummyEvaluator(Evaluator):
    def evaluate(self, image: Image.Image) -> Score:
        return Score(0.0)
