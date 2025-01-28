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


if __name__ == "__main__":
    import sys
    pytest.main(sys.argv)
