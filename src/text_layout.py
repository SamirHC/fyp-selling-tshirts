from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum, auto

from PIL import Image

from src.render_text import render_text
from src.text_splitter import TextSplitter, WordIndexSplit


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
    def render(text, **kwargs):
        pass


class Identity(TextLayout):
    render_required_kwargs = {
        "font_path",
        "font_size",
        "text_color",
    }

    @staticmethod
    def render(text, **kwargs):
        font_path = kwargs["font_path"]
        font_size = kwargs["font_size"]
        text_color = kwargs["text_color"]

        return render_text(text, font_path, font_size, text_color, return_bbox=True)


class Multiline(TextLayout):
    render_required_kwargs = {
        "text_layout_list",
        "text_layout_kwargs_list",
        "align",
        "text_splitter",
        "line_spacing",
    }

    @staticmethod
    def render(text, **kwargs) -> Image.Image:
        text_layout_list: list[TextLayout] = kwargs["text_layout_list"]
        text_layout_kwargs_list = kwargs["text_layout_kwargs_list"]
        align: Align = kwargs["align"]
        text_splitter: TextSplitter = kwargs["text_splitter"]
        line_spacing = kwargs["line_spacing"]

        lines = text_splitter.split(text)

        imgs = []
        bboxs = []
        for line, layout, kws in zip(lines, text_layout_list, text_layout_kwargs_list, strict=True):
            img, bbox = layout.render(line, **kws)
            imgs.append(img)
            bboxs.append(bbox)

        left = min(bbox[0] for bbox in bboxs)
        top = bboxs[0][1]

        width = max(im.size[0] for im in imgs)
        height = sum(line_spacing[:-1]) + bboxs[-1][1] - top + imgs[-1].size[1]
        result = Image.new("RGBA", (width, height), TRANSPARENT)
        overall_bbox = (left, top, left + width, top + height)

        y = 0
        for im, bbox, dy in zip(imgs, bboxs, line_spacing):
            paste_x = -left + bbox[0] + align.get_x(width, im.size[0])
            paste_y = -top + bbox[1] + y
            result.paste(im, (paste_x, paste_y), im)
            y += dy

        return result, overall_bbox


if __name__ == "__main__":
    import os

    fonts = [
        os.path.join("data", "fonts", "en", "great-vibes", "GreatVibes-Regular.ttf"),
        os.path.join("data", "fonts", "en", "adler", "Adler.ttf"),
        os.path.join("data", "fonts", "en", "aladin", "Aladin-Regular.ttf"),
        os.path.join("data", "fonts", "en", "aleo", "Aleo-Regular.otf"),
    ]

    image, _ = Multiline.render(
        "Earth is my favourite planet",
        text_layout_list=[Identity] * 4,
        text_layout_kwargs_list=[
            {
                "font_path": fonts[0],
                "font_size": 144,
                "text_color": (200, 0, 0)
            },
            {
                "font_path": fonts[1],
                "font_size": 36,
                "text_color": (130, 0, 0)
            },
            {
                "font_path": fonts[2],
                "font_size": 144,
                "text_color": (255, 0, 0)
            },
            {
                "font_path": fonts[3],
                "font_size": 72,
                "text_color": (220, 0, 0)
            },
        ],
        align=Align.CENTER,
        text_splitter=WordIndexSplit([0,1,3,4,5]),
        line_spacing=[135, -20, 120, 0],
    )

    image.show()
