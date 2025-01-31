import pytest

from src.design_generation import internal_repr as ir


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
                size=(100, 100),
                position=(0, 0),
                components=[]
            ),
        ]
    ).to_dict()
    expected = {
        "canvas_size": (500, 500),
        "layers": [
        {
            "size": (100, 100),
            "position": (0, 0),
            "components": [], 
        },
    ]}

    assert out == expected


def test_empty_layer_to_dict():
    out = ir.Layer(
        size=(100, 100),
        position=(0, 0),
        components=[]
    ).to_dict()
    expected = {
        "size": (100, 100),
        "position": (0, 0),
        "components": [], 
    }

    assert out == expected


def test_text_component_to_dict():
    out = ir.TextComponent(
        position=(0, 0),
        text="Hello World",
        layout="Identity",
        font_path="fake/font/path",
        font_size=36,
        text_color=(130, 0, 0)
    ).to_dict()
    expected = {
        "position": (0, 0),
        "text": "Hello World",
        "layout": "Identity",
        "layout_kwargs": {
            "font_path": "fake/font/path",
            "font_size": 36,
            "text_color": (130, 0, 0),
        },
    }

    assert out == expected


if __name__ == "__main__":
    import sys
    pytest.main(sys.argv)
