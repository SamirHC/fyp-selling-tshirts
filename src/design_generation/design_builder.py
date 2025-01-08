from __future__ import annotations
from typing import Type

from PIL import Image

from src.design_generation import text_layout


class DesignBuilder:
    def __init__(self, width=500, height=500):
        self.width = width
        self.height = height
        self.components = []
    
    def set_size(self, width, height) -> DesignBuilder:
        self.width = width
        self.height = height
        return self

    def add_component(self, image: Image.Image, position: tuple[int, int]) -> DesignBuilder:
        self.components.append({
            "image": image,
            "x": position[0],
            "y": position[1]
        })
        return self


class TextComponentBuilder:
    def __init__(self):
        self.text = ""
        self.layout = text_layout.Identity
        self.layout_params = {}
    
    def set_text_components(self, text_components) -> TextComponentBuilder:
        self.text_components = text_components
        return self
    
    def set_layout(self, layout: Type[text_layout.TextLayout]) -> TextComponentBuilder:
        self.layout = layout
        return self

    def set_layout_param(self, param: str, value) -> TextComponentBuilder:
        self.layout_params[param] = value
        return self
    
    def set_layout_kwargs(self, kwargs) -> TextComponentBuilder:
        self.layout_params = kwargs
        return self

    def build(self) -> Image.Image:
        return self.layout.render(self.text_components, **self.layout_params)
