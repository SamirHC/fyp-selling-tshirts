from PIL import Image
import pytest

from src.design_generation import image_edit


@pytest.fixture(scope="session")
def red_blue_image():
    image = Image.new("RGB", (100, 100), "red")
    object = Image.new("RGB", (50, 50), "blue")
    image.paste(object, (25, 25))
    return image


def test_remove_bg_contains_transparency():
    image = Image.new("RGB", (100, 100), "red")
    object = Image.new("RGB", (50, 50), "blue")
    image.paste(object, (25, 25))

    edited_image = image_edit.remove_bg(image)

    assert 0 in list(edited_image.getdata(3))


def test_crop_to_content_removes_transparent_background():
    image = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    object = Image.new("RGBA", (50, 50), "blue")
    image.paste(object, (25, 25))

    edited_image = image_edit.crop_to_content(image)

    assert edited_image == object


def test_crop_to_content_does_not_remove_translucent_background():
    image = Image.new("RGBA", (100, 100), (0, 0, 0, 128))
    object = Image.new("RGBA", (50, 50), "blue")
    image.paste(object, (25, 25))

    edited_image = image_edit.crop_to_content(image)

    assert edited_image.size == (100, 100)


if __name__ == "__main__":
    import sys
    pytest.main(sys.argv)
