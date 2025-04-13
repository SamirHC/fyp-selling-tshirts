from PIL import Image
import pytest

from src.common import utils


def test_get_image_from_url_returns_image():
    image = utils.get_image_from_url("https://placehold.co/600x400.png")

    assert isinstance(image, Image.Image)


def test_image_to_base64_returns_str():
    image = Image.new("RGB", (10, 10))
    base64 = utils.image_to_base64(image)

    assert isinstance(base64, str)


if __name__ == "__main__":
    import sys
    pytest.main(sys.argv)
