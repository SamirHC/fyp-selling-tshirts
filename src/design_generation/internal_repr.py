from abc import ABC, abstractmethod
import base64
import io
from PIL import Image


class Node(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def to_dict(self):
        return {}


class Component(Node):
    pass


class ImageComponent(Component):
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


class Layer(Node):
    def __init__(self, size, position, components: list[Component]):
        self.size = size
        self.position = position
        self.components = components

    def to_dict(self):
        return {
            "size": self.size,
            "position": self.position,
            "components": [component.to_dict() for component in self.components],
        }


class Design(Node):
    def __init__(self, canvas_size, layers: list[Layer]):
        self.canvas_size = canvas_size
        self.layers = layers

    def to_dict(self):
        return {
            "canvas_size": self.canvas_size,
            "layers": [layer.to_dict() for layer in self.layers]
        }


def construct_from_ir(design_ir: Design):
    image = Image.new("RGB", design_ir.canvas_size)
    
    for layer_ir in design.layers:
        layer = Image.new("RGB", layer_ir.size)
        for component_ir in layer_ir.components:
            if isinstance(component_ir, ImageComponent):
                component_image = Image.open(
                    io.BytesIO(base64.b64decode(component_ir.base64_image))
                )
                layer.paste(component_image, component_ir.position)
            else:
                pass

        image.paste(layer, layer_ir.position)

    image.show()
    return image


if __name__ == "__main__":
    design = Design(
        canvas_size=(500, 500),
        layers=[
            Layer(
                size=(300, 200),
                position=(100, 200),
                components=[
                    ImageComponent(Image.new("RGB", (100, 100), "red"), (0, 0)),
                ],
            ),
        ]
    )

    print(design.to_dict())
    construct_from_ir(design)
