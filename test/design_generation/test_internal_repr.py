from lxml import etree
from PIL import Image
import pytest

from src.design_generation import internal_repr as ir


################################################################################
# ImageComponent
################################################################################

@pytest.fixture(scope="session")
def image_component() -> ir.ImageComponent:
    image = Image.new("RGB", (50, 50), (255, 0, 0))

    return ir.ImageComponent(
        image=image,
        position=(30, 40)
    )


def test_image_component_to_svg(image_component: ir.ImageComponent):
    svg = image_component.to_svg()

    assert isinstance(svg, str)


def test_image_component_to_lxml(image_component: ir.ImageComponent):
    lxml = image_component.to_lxml()

    assert isinstance(lxml, etree._Element)


def test_image_component_get_dependencies(image_component: ir.ImageComponent):
    ds = image_component.get_dependencies()

    assert ds == set()


################################################################################
# TextComponent
################################################################################

@pytest.fixture(scope="session")
def text_component() -> ir.TextComponent:
    return ir.TextComponent(
        position=(0, 0),
        text="Hello World",
        font_family="family_name",
        font_size=36,
        fill=(130, 0, 0)
    )


def test_text_component_to_dict():
    out = ir.TextComponent(
        position=(0, 0),
        text="Hello World",
        font_family="family_name",
        font_size=36,
        fill=(130, 0, 0)
    ).to_dict()
    expected = {
        "position": (0, 0),
        "text": "Hello World",
        "font_family": "family_name",
        "font_size": 36,
        "fill": (130, 0, 0),
    }

    assert out == expected


def test_text_component_to_svg(text_component: ir.TextComponent):
    svg = text_component.to_svg()

    assert isinstance(svg, str)


def test_text_component_to_lxml(text_component: ir.TextComponent):
    lxml = text_component.to_lxml()

    assert isinstance(lxml, etree._Element)


def test_text_component_get_dependencies(text_component: ir.TextComponent):
    ds = text_component.get_dependencies()

    assert len(ds) == 1
    assert ds == {("font-family", "family_name")}


################################################################################
# Layer
################################################################################

@pytest.fixture(scope="session")
def empty_layer() -> ir.Layer:
    return ir.Layer(
        position=(0, 0),
        components=[]
    )


@pytest.fixture(scope="session")
def text_image_layer(text_component, image_component) -> ir.Layer:
    return ir.Layer(
        position=(0, 0),
        components=[text_component, image_component]
    )

@pytest.fixture(scope="session")
def multi_dependency_layer() -> ir.Layer:
    c1 = ir.TextComponent(
        position=(0, 0),
        text="Hello",
        font_family="hello-font",
        font_size=36,
        fill=(130, 0, 0)
    )
    c2 = ir.TextComponent(
        position=(0, 50),
        text="World",
        font_family="world-font",
        font_size=36,
        fill=(130, 0, 0)
    )
    c3 = ir.TextComponent(
        position=(0, 100),
        text="!!!!",
        font_family="world-font",
        font_size=36,
        fill=(130, 0, 0)
    )
    return ir.Layer(
        position=(0, 0),
        components=[c1, c2, c3]
    )


def test_empty_layer_to_dict():
    out = ir.Layer(
        position=(0, 0),
        components=[]
    ).to_dict()
    expected = {
        "position": (0, 0),
        "components": [], 
    }

    assert out == expected


def test_layer_to_svg(empty_layer: ir.Layer, text_image_layer: ir.Layer, text_component, image_component):
    empty_layer_svg = empty_layer.to_svg()
    text_image_layer_svg = text_image_layer.to_svg()

    assert isinstance(empty_layer_svg, str)
    assert isinstance(text_image_layer_svg, str)

    assert text_component.to_svg() in text_image_layer_svg
    assert image_component.to_svg() in text_image_layer_svg


def test_layer_to_lxml(empty_layer: ir.Layer):
    lxml = empty_layer.to_lxml()

    assert isinstance(lxml, etree._Element)


def test_layer_get_dependencies(empty_layer: ir.Layer, text_image_layer: ir.Layer):
    empty_layer_ds = empty_layer.get_dependencies()
    text_image_layer_ds = text_image_layer.get_dependencies()

    assert empty_layer_ds == set()
    assert text_image_layer_ds == {("font-family", "family_name")}


def test_layer_get_multiple_dependencies(multi_dependency_layer: ir.Layer):
    ds = multi_dependency_layer.get_dependencies()

    assert ds == {("font-family", "hello-font"), ("font-family", "world-font")}


################################################################################
# Design
################################################################################

@pytest.fixture(scope="session")
def empty_design() -> ir.Design:
    return ir.Design(
        canvas_size=(500, 500),
        layers=[]
    )


@pytest.fixture(scope="session")
def complex_design(multi_dependency_layer: ir.Layer) -> ir.Design:
    return ir.Design(
        canvas_size=(500, 500),
        layers=[multi_dependency_layer, multi_dependency_layer]
    )


def test_empty_design_to_dict():
    out = ir.Design(
        canvas_size=(500, 500),
        layers=[]
    ).to_dict()
    expected = {"canvas_size": (500, 500), "layers": []}

    assert out == expected


def test_empty_layer_design_to_dict():
    out = ir.Design(
        canvas_size=(500, 500),
        layers=[
            ir.Layer(
                position=(0, 0),
                components=[]
            ),
        ]
    ).to_dict()
    expected = {
        "canvas_size": (500, 500),
        "layers": [
        {
            "position": (0, 0),
            "components": [], 
        },
    ]}

    assert out == expected


def test_design_to_lxml(empty_design: ir.Design):
    lxml = empty_design.to_lxml()

    assert isinstance(lxml, etree._Element)


def test_design_to_svg(empty_design: ir.Design):
    svg = empty_design.to_svg()

    assert isinstance(svg, str)


def test_design_get_dependencies(complex_design: ir.Design):
    ds = complex_design.get_dependencies()

    assert ds == {("font-family", "hello-font"), ("font-family", "world-font")}


if __name__ == "__main__":
    import sys
    pytest.main(sys.argv)
