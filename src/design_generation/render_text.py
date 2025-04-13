from PIL import Image, ImageDraw, ImageFont

from src.common import constants


def render_text(text: str, font_path: str, font_size=72, text_color=constants.Color.BLACK, return_bbox=False):
    font = ImageFont.truetype(font_path, size=font_size)
    bbox = font.getbbox(text)
    left, top, right, bottom = bbox
    offset = (-left, -top)
    width = right - left
    height = bottom - top

    text_image = Image.new("RGBA", (width, height), constants.Color.TRANSPARENT)
    draw = ImageDraw.Draw(text_image)
    draw.text(offset, text, fill=text_color, font=font)

    if return_bbox:
        return text_image, bbox
    else:
        return text_image


if __name__ == "__main__":
    import os

    font_path = os.path.join("data", "fonts", "en", "great-vibes", "GreatVibes-Regular.ttf")
    text_image = render_text(
        "Earth is my favourite planet",
        font_path=font_path,
        text_color=(255, 0, 0)
    )
    text_image.show()
