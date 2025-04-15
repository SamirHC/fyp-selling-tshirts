from abc import ABC, abstractmethod
import base64
from PIL import Image
from lxml import etree

from src.common import utils
from src.data_collection import fonts
from src.design_generation import render_text


font_df = fonts.get_font_data()


class Node(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass

    @abstractmethod
    def to_lxml(self) -> etree._Element:
        pass

    def to_svg(self) -> str:
        return etree.tostring(self.to_lxml(), pretty_print=True, encoding="utf-8").decode()

    def get_dependencies(self):
        return set()


class ImageComponent(Node):
    def __init__(self, image: Image.Image, position):
        self.base64_image = utils.image_to_base64(image)
        self.position = position

    def to_dict(self):
        return {
            "position": self.position,
            "base64_image": self.base64_image
        }

    def to_lxml(self):
        return etree.Element(
            "image",
            x=str(self.position[0]),
            y=str(self.position[1]),
            href=f"data:image/png;base64,{self.base64_image}"
        )


class TextComponent(Node):
    def __init__(self, position, text, font_family, font_size, fill, tag="text"):
        self.position = position
        self.text = text
        self.font_family = font_family
        self.font_size = font_size
        self.fill = fill
        self.tag = tag
        self.dependencies = set()
        if tag == "text":
            self.dependencies.add(("font-family", self.font_family))

    def to_dict(self):
        return {
            "position": self.position,
            "text": self.text,
            "font_family": self.font_family,
            "font_size": self.font_size,
            "fill": self.fill
        }

    def to_lxml(self):
        match self.tag:
            case "text":
                svg = etree.Element(
                    "text",
                    attrib={
                        "x": str(self.position[0]),
                        "y": str(self.position[1]),
                        "font-family": self.font_family,
                        "font-size": str(self.font_size),
                        "fill": f"rgb{self.fill}",
                        "text-anchor": "middle",
                        "dominant-baseline": "middle",
                    }
                )
                svg.text = self.text
            case "image":
                image, bbox = render_text.render_text(
                    self.text,
                    font_path=font_df[font_df["family"] == self.font_family].iloc[0]["path"],
                    font_size=self.font_size,
                    text_color=self.fill,
                    return_bbox=True
                )
                base64_image = utils.image_to_base64(image)
                svg = etree.Element(
                    "image",
                    x=str(self.position[0]),
                    y=str(self.position[1]-bbox[3]+bbox[1]),
                    href=f"data:image/png;base64,{base64_image}"
                )
        return svg

    def get_dependencies(self):
        return self.dependencies


class Layer(Node):
    def __init__(self, position, components: list[Node]):
        self.position = position
        self.components = components

    def to_dict(self):
        return {
            "position": self.position,
            "components": [component.to_dict() for component in self.components],
        }

    def to_lxml(self):
        svg: etree._Element = etree.Element(
            "g",
            transform=f"translate{self.position}",
        )
        svg.extend(component.to_lxml() for component in self.components)
        return svg
    
    def get_dependencies(self):
        dependencies = set()
        for component in self.components:
            dependencies = dependencies.union(component.get_dependencies())

        return dependencies


class Design(Node):
    def __init__(self, canvas_size, layers: list[Layer]):
        self.canvas_size = canvas_size
        self.layers = layers

    def to_dict(self):
        return {
            "canvas_size": self.canvas_size,
            "layers": [layer.to_dict() for layer in self.layers]
        }

    def to_lxml(self):
        svg: etree._Element = etree.Element(
            "svg",
            width=str(self.canvas_size[0]),
            height=str(self.canvas_size[1]),
            xmlns="http://www.w3.org/2000/svg",
        )
        dependencies = self.get_dependencies()
        if dependencies:
            defs = etree.Element("defs")
            svg.append(defs)
            for dependency in dependencies:
                match dependency:
                    case ("font-family", font_family):
                        # Gets the first font row matching the family
                        font_data = font_df[font_df["family"] == font_family].iloc[0]
                        with open(font_data["path"], "rb") as font_file:
                            font_base64_encoded = base64.b64encode(font_file.read()).decode("utf-8")
                        font_el = etree.SubElement(defs, "style", type="text/css")
                        font_el.text = (
                            "\n"
                            "@font-face {\n"
                            f"  font-family: \"{font_family}\";\n"
                            f"  src: url(\"data:font/{font_data["format"]};base64,{font_base64_encoded}\") format(\"{font_data["format"]}\");\n"
                            "}\n"
                        )

        svg.extend(layer.to_lxml() for layer in self.layers)
        
        svg.append(
            etree.Element("rect", attrib={
                "x": "0",
                "y": "0",
                "width": str(self.canvas_size[0]),
                "height": str(self.canvas_size[1]),
                "fill": "none",
                "stroke": "black",
                "stroke-width": "1",
            })
        )
        return svg

    def get_dependencies(self):
        dependencies = set()
        for layer in self.layers:
            dependencies = dependencies.union(layer.get_dependencies())

        return dependencies


if __name__ == "__main__":
    import os
    import webbrowser
    from src.data_collection import fonts

    font_data = fonts.get_font_data()
    font_paths = list(font_data["path"])

    design = Design(
        canvas_size=(500, 500),
        layers=[
            Layer(
                position=(100, 200),
                components=[
                    ImageComponent(
                        image=Image.new("RGB", (100, 100), "red"),
                        position=(0, 0)
                    ),
                ],
            ),
            Layer(
                position=(200, 200),
                components=[
                    TextComponent(
                        position=(0, 50),
                        text="Hello World",
                        font_family="cookie",
                        font_size=36,
                        fill=(130, 0, 0)
                    )
                ]
            )
        ]
    )

    svg = design.to_svg()
    print(svg)

    svg_path = os.path.join("out", "internal_repr.svg")
    with open(svg_path, "w") as f:
        f.write(svg)

    webbrowser.get("firefox").open(svg_path)
