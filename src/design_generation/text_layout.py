from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum, auto

from PIL import Image

from src.design_generation.render_text import render_text


BLACK = (0, 0, 0, 255)
TRANSPARENT = (0, 0, 0, 0)


class Align(Enum):
    LEFT = auto()
    CENTER = auto()
    RIGHT = auto()

    def get_x(self, region_width: int, object_width: int) -> int:
        match self:
            case Align.LEFT:
                return 0
            case Align.RIGHT:
                return region_width - object_width
            case Align.CENTER:
                return (region_width - object_width) // 2


class TextLayout(ABC):
    __slots__ = ("render_required_kwargs",)

    @staticmethod
    @abstractmethod
    def render(text_components, **kwargs):
        pass


class Identity(TextLayout):
    render_required_kwargs = {
        "font_path",
        "font_size",
        "text_color",
    }

    @staticmethod
    def render(text_components: str, **kwargs) -> tuple[Image.Image, tuple[float, float, float, float]]:
        font_path = kwargs["font_path"]
        font_size = kwargs["font_size"]
        text_color = kwargs["text_color"]

        return render_text(text_components, font_path, font_size, text_color, return_bbox=True)


class Multiline(TextLayout):
    render_required_kwargs = {
        "align",
        "line_spacing",
    }

    @staticmethod
    def render(text_components: list[tuple[Image.Image, tuple[float, float, float, float]]], **kwargs) -> tuple[Image.Image, tuple[float, float, float, float]]:
        align: Align = kwargs["align"]

        line_spacing = kwargs["line_spacing"]
        if isinstance(line_spacing, int):
            line_spacing = [line_spacing]
        line_spacing += [0] * (len(text_components) - len(line_spacing))

        imgs, bboxs = zip(*text_components, strict=True)

        left = min(bbox[0] for bbox in bboxs)
        top = bboxs[0][1]

        width = max(im.size[0] for im in imgs)
        height = sum(line_spacing) + bboxs[-1][1] - top + imgs[-1].size[1]
        result = Image.new("RGBA", (width, height), TRANSPARENT)
        overall_bbox = (left, top, left + width, top + height)

        y = 0
        for im, bbox, dy in zip(imgs, bboxs, line_spacing):
            paste_x = -left + bbox[0] + align.get_x(width, im.size[0])
            paste_y = -top + bbox[1] + y
            result.paste(im, (paste_x, paste_y), im)
            y += dy

        return result, overall_bbox


def get_text_layouts():
    return [
        Identity,
        Multiline,
    ]


if __name__ == "__main__":
    from src.design_generation.text_splitter import WordIndexSplit
    from src.design_generation import fonts


    font_data = fonts.get_font_data()
    font_paths = list(font_data["path"])

    text = WordIndexSplit([0,1,3,4,5]).split("Earth is my favourite planet")

    text_layout_kwargs_list = [
        {
            "font_path": font_paths[0],
            "font_size": 144,
            "text_color": (200, 0, 0)
        },
        {
            "font_path": font_paths[1],
            "font_size": 36,
            "text_color": (130, 0, 0)
        },
        {
            "font_path": font_paths[2],
            "font_size": 144,
            "text_color": (255, 0, 0)
        },
        {
            "font_path": font_paths[3],
            "font_size": 72,
            "text_color": (220, 0, 0)
        }
    ]
    text_components = [
        Identity.render(t, **kwargs)
        for t, kwargs in zip(text, text_layout_kwargs_list)
    ]

    image, _ = Multiline.render(
        text_components,
        align=Align.CENTER,
        line_spacing=[135, -20, 120],
    )

    image.show()
