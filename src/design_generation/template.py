from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

import src.design_generation.internal_repr as ir


class Template(ABC):
    @property
    @abstractmethod
    def design(self) -> ir.Design:
        pass

    def save_svg(self, file_path):
        xml = self.to_svg()

        with open(file_path, "w+") as f:
            f.write(xml)

    def to_lxml(self):
        return self.design.to_lxml()

    def to_svg(self):
        return self.design.to_svg()


class SimpleText(Template):
    def __init__(self, canvas_size, font, font_size, color, text):
        self.canvas_size = canvas_size
        self.font = font
        self.font_size = font_size
        self.color = color
        self.text = text

    @property
    def design(self):
        components = []
        text_component = ir.TextComponent(
            position=(self.canvas_size[0]//2, self.canvas_size[1]//4),
            text=self.text,
            font_family=self.font,
            font_size=self.font_size,
            fill=self.color,
        )
        components.append(text_component)

        layer = ir.Layer(
            position=(0, 0),
            components=components
        )
        return ir.Design(self.canvas_size, [layer])


class SimpleMultilineText(Template):
    def __init__(self, canvas_size, font, font_size, color, text_lines):
        self.canvas_size = canvas_size
        self.font = font
        self.font_size = font_size
        self.color = color
        self.text_lines = text_lines

    @property
    def design(self):
        components = []
        for i, text_line in enumerate(self.text_lines):
            dy = i * self.font_size * 1.2
            components.append(
                ir.TextComponent(
                    position=(self.canvas_size[0]//2, self.canvas_size[1]//4 + dy),
                    text=text_line,
                    font_family=self.font,
                    font_size=self.font_size,
                    fill=self.color,
                )
            )

        layer = ir.Layer(
            position=(0, 0),
            components=components
        )
        return ir.Design(self.canvas_size, [layer])


class CenterImage(Template):
    def __init__(self, canvas_size, image: Image.Image):
        self.canvas_size = canvas_size
        self.image = image

    @property
    def design(self):
        components = []
        image_component = ir.ImageComponent(
            image=self.image,
            position=(
                self.canvas_size[0]//2 - self.image.width//2,
                self.canvas_size[1]//2 - self.image.height//2
            ),
        )
        components.append(image_component)

        layer = ir.Layer(
            position=(0, 0),
            components=components
        )


        layer = ir.Layer(
            position=(0, 0),
            components=components
        )
        return ir.Design(self.canvas_size, [layer])


class CaptionedImage(Template):
    def __init__(self, canvas_size, image: Image.Image, font, font_size, color, top_text, bottom_text):
        self.canvas_size = canvas_size
        self.image = image
        self.font = font
        self.font_size = font_size
        self.color = color
        self.top_text = top_text
        self.bottom_text = bottom_text

    @property
    def design(self):
        image_components = []
        image = self.image.resize((int(self.canvas_size[0]*0.8), int(self.canvas_size[1]*0.8)))
        image_component = ir.ImageComponent(
            image=image,
            position=(
                self.canvas_size[0]//2 - image.width//2,
                self.canvas_size[1]//2 - image.height//2
            ),
        )
        image_components.append(image_component)
        image_layer = ir.Layer(
            position=(0, 0),
            components=image_components
        )

        text_components = []
        if self.top_text:
            top_text_component = ir.TextComponent(
                position=(self.canvas_size[0]//2, self.canvas_size[1]*0.1),
                text=self.top_text,
                font_family=self.font,
                font_size=self.font_size,
                fill=self.color,
            )
            text_components.append(top_text_component)

        if self.bottom_text:
            bottom_text_component = ir.TextComponent(
                position=(self.canvas_size[0]//2, self.canvas_size[1]*0.9),
                text=self.bottom_text,
                font_family=self.font,
                font_size=self.font_size,
                fill=self.color,
            )
            text_components.append(bottom_text_component)

        text_layer = ir.Layer(
            position=(0, 0),
            components=text_components
        )
        return ir.Design(self.canvas_size, [image_layer, text_layer])


class GridCollage(Template):
    def __init__(self, canvas_size, images: list[Image.Image], grid_x, grid_y):
        assert len(images) == grid_x * grid_y
        self.canvas_size = canvas_size
        self.images = images
        self.grid_x = grid_x
        self.grid_y = grid_y

    @property
    def design(self):
        image_components = []
        xs = np.linspace(0, self.canvas_size[0], self.grid_x + 1)
        ys = np.linspace(0, self.canvas_size[1], self.grid_y + 1)
        for i, image in enumerate(self.images):
            position = (xs[i % self.grid_x], ys[i // self.grid_x])
            image = image.resize((self.canvas_size[0] // self.grid_x, self.canvas_size[1] // self.grid_y))
            image_components.append(ir.ImageComponent(image, position))

        image_layer = ir.Layer(
            position=(0, 0),
            components=image_components
        )
        return ir.Design(self.canvas_size, [image_layer])


if __name__ == "__main__":
    import os

    save_path = os.path.join("out", "template_test_1.svg")
    t1 = SimpleText((500, 500), "cookie", 72, (0, 0, 0), "Simple & Bold")
    t1.save_svg(save_path)

    save_path = os.path.join("out", "template_test_2.svg")
    t2 = SimpleMultilineText((500, 500), "barrio", 60, (52, 102, 79), ["Earth", "is my", "favourite", "planet"])
    t2.save_svg(save_path)

    save_path = os.path.join("out", "template_test_3.svg")
    image = Image.new("RGB", (100, 100), "red")
    object = Image.new("RGB", (50, 50), "blue")
    image.paste(object, (25, 25))
    t3 = CenterImage((500, 500), image)
    t3.save_svg(save_path)

    save_path = os.path.join("out", "template_test_4.svg")
    image = Image.new("RGB", (200, 200), (255, 245, 245))
    object = Image.new("RGB", (100, 100), "yellow")
    image.paste(object, (50, 50))
    t4 = CaptionedImage((500, 500), image, "playball", 92, (255, 215, 0), "I Love", "Eggs")
    t4.save_svg(save_path)

    save_path = os.path.join("out", "template_test_5.svg")
    red_square = Image.new("RGB", (100, 100), "red")
    blue_square = Image.new("RGB", (100, 100), "blue")
    t5 = GridCollage(
        canvas_size=(500, 500),
        images=[red_square, blue_square, blue_square, red_square],
        grid_x=2,
        grid_y=2,
    )
    t5.save_svg(save_path)
