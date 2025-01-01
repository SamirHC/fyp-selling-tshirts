import os

from PIL import Image, ImageDraw, ImageFont


# Colors
BLACK = (0, 0, 0, 255)
TRANSPARENT = (0, 0, 0, 0)


def render_text(text: str, font: ImageFont.FreeTypeFont, text_color=BLACK):
    bbox = font.getbbox(text)
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top

    text_image = Image.new("RGBA", (width, height), TRANSPARENT)
    draw = ImageDraw.Draw(text_image)
    draw.text((-left, -top), text, fill=text_color, font=font)

    return text_image


if __name__ == "__main__":
    font_path = os.path.join("data", "fonts", "great-vibes", "GreatVibes-Regular.ttf")
    text_image = render_text(
        "Hello World!",
        font=ImageFont.truetype(font_path, size=72),
        text_color=(255, 0, 0)
    )
    text_image.show()
