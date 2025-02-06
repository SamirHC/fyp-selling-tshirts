from abc import ABC, abstractmethod
import base64
import io
from PIL import Image
from lxml import etree


class Node(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def to_dict(self):
        return

    @abstractmethod
    def to_svg(self):
        return


class ImageComponent(Node):
    def __init__(self, image: Image.Image, position):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        self.base64_image = base64.b64encode(buffer.getvalue())
        self.position = position

    def to_dict(self):
        return {
            "position": self.position,
            "base64_image": self.base64_image
        }

    def to_svg(self):
        return etree.Element(
            "image",
            x=str(self.position[0]),
            y=str(self.position[1]),
            href=f"data:image/png;base64,{self.base64_image.decode("utf-8")}"
        )


class TextComponent(Node):
    def __init__(self, position, text, font_path, font_size, fill):
        self.position = position
        self.text = text
        self.font_path = font_path
        self.font_size = font_size
        self.fill = fill

    def to_dict(self):
        return {
            "position": self.position,
            "text": self.text,
            "font_path": self.font_path,
            "font_size": self.font_size,
            "fill": self.fill
        }

    def to_svg(self):
        svg = etree.Element(
            "text",
            attrib={
                "x": str(self.position[0]),
                "y": str(self.position[1]),
                "font-family": "Arial",
                "font-size": str(40),
                "fill": "black",
            }
        )
        svg.text = self.text
        return svg


class Layer(Node):
    def __init__(self, size, position, components: list[Node]):
        self.size = size
        self.position = position
        self.components = components

    def to_dict(self):
        return {
            "size": self.size,
            "position": self.position,
            "components": [component.to_dict() for component in self.components],
        }

    def to_svg(self):
        svg: etree._Element = etree.Element(
            "svg",
            x=str(self.position[0]),
            y=str(self.position[1]),
            width=str(self.size[0]),
            height=str(self.size[1]),
        )
        svg.extend(component.to_svg() for component in self.components)
        return svg


class Design(Node):
    def __init__(self, canvas_size, layers: list[Layer]):
        self.canvas_size = canvas_size
        self.layers = layers

    def to_dict(self):
        return {
            "canvas_size": self.canvas_size,
            "layers": [layer.to_dict() for layer in self.layers]
        }

    def to_svg(self):
        svg: etree._Element = etree.Element(
            "svg",
            width=str(self.canvas_size[0]),
            height=str(self.canvas_size[1]),
            xmlns="http://www.w3.org/2000/svg",
        )
        svg.extend(layer.to_svg() for layer in self.layers)
        return svg


if __name__ == "__main__":
    from src.data_collection import fonts

    font_data = fonts.get_font_data()
    font_paths = list(font_data["path"])

    design = Design(
        canvas_size=(500, 500),
        layers=[
            Layer(
                size=(300, 200),
                position=(100, 200),
                components=[
                    ImageComponent(
                        image=Image.new("RGB", (100, 100), "red"),
                        position=(0, 0)
                    ),
                ],
            ),
            Layer(
                size=(300, 100),
                position=(200, 200),
                components=[
                    TextComponent(
                        position=(0, 0),
                        text="Hello World",
                        font_path=font_paths[1],
                        font_size=36,
                        fill=(130, 0, 0)
                    )
                ]
            )
        ]
    )

    print(design.to_dict())

    print()
    xml = etree.tostring(design.to_svg(), pretty_print=True)
    print(xml.decode(), end='')
